# Opening books

The app reads standard Polyglot `.bin` opening books. Its default local file is
`books/gm2001.bin`; downloaded binaries are ignored by Git.

The current local copy came from the `gm2001.zip` asset in the
[polyglot-books v1.0 release](https://github.com/ChrisWhittington/polyglot-books/releases/tag/v1.0).
That collection credits Oliver Deville and describes the source games as GM games
from 2001–2013 with a 2530+ threshold. The collection says the books remain the
property of their respective authors, so this project does not claim authorship
or a more permissive license.

Verified download:

- Archive: `gm2001.zip`, 302,200 bytes
- Archive SHA-256: `68514dcedac0394dbda5b4cae1a834f274e1b1e7fde5dce310c107924c03aa6a`
- Extracted file: `gm2001.bin`, 486,656 bytes
- Extracted SHA-256: `fb6e9f3f27bb19a5b2fdefcc441c88ddeae48db61d5f00ad83973abb9f939c87`

Structural inspection found 30,416 position–move entries and 23,813 distinct
stored position keys. Starting from normal chess and following every legal stored
move reaches book continuations through ply 30 (15 full moves). The app deliberately
caps lookup at ply 20 (10 full moves). A line may leave the book earlier when it
reaches an unrecorded position.

A seeded 10,000-line weighted simulation reached at least 10 book plies in 93.9%
of lines and the configured 20-ply cap in 49.6%; mean coverage was 17.1 plies and
the median was 19. These figures describe this book's own weighted choices. A human
opponent choosing rare deviations can force an earlier exit, so they are not a
guarantee of game coverage.

To replace it, put another Polyglot book at that path or set `CHESS_BOOK_PATH` to
an absolute `.bin` path before starting the backend. Book support is read-only.
Use `"use_book": false` in API requests for controlled engine comparisons.
