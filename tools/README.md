# Local benchmark opponent

`stockfish-sf19/` contains the unmodified official Stockfish 19 macOS universal
release, downloaded for local testing. It is excluded from version control.
The bundle retains its upstream license, authors, documentation, and source.

Official release: https://github.com/official-stockfish/Stockfish/releases/tag/sf_19

Archive: https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-macos-universal.tar.gz

Verified archive SHA-256 (also supplied by the GitHub release API):
`a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77`

Executable: `tools/stockfish-sf19/stockfish/stockfish-macos-universal`.
On another platform, obtain the matching official build and pass its path to
`--stockfish`. Match reports record the binary hash and UCI identification.
Stockfish is used as an external opponent and reviewer; the application still
uses this project's own search and heuristic evaluation.
