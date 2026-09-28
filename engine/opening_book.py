"""Polyglot opening-book lookup with explicit, inspectable metadata."""
from dataclasses import dataclass
from pathlib import Path
import random

import chess
import chess.polyglot


class OpeningBookError(ValueError):
    """The configured book cannot be read as a Polyglot book."""


@dataclass(frozen=True)
class BookChoice:
    move: chess.Move
    weight: int
    alternatives: int


def choose_book_move(board: chess.Board, path, *, rng=None) -> BookChoice | None:
    """Choose a legal move by Polyglot weight, or return None when out of book."""
    book_path = Path(path)
    if not book_path.is_file():
        raise OpeningBookError(f"opening book not found: {book_path}")
    if book_path.stat().st_size % 16:
        raise OpeningBookError("Polyglot book size must be a multiple of 16 bytes")

    try:
        with chess.polyglot.open_reader(book_path) as reader:
            entries = [entry for entry in reader.find_all(board)
                       if entry.move in board.legal_moves and entry.weight > 0]
    except (OSError, ValueError) as exc:
        raise OpeningBookError(f"could not read opening book: {exc}") from exc

    if not entries:
        return None
    random_source = random if rng is None else rng
    total = sum(entry.weight for entry in entries)
    pick = random_source.randrange(total)
    for entry in entries:
        if pick < entry.weight:
            return BookChoice(entry.move, entry.weight, len(entries))
        pick -= entry.weight
    raise AssertionError("weighted book selection exhausted its range")


def inspect_book(path) -> dict:
    """Return structural counts without claiming every stored key is reachable."""
    book_path = Path(path)
    if not book_path.is_file():
        raise OpeningBookError(f"opening book not found: {book_path}")
    if book_path.stat().st_size % 16:
        raise OpeningBookError("Polyglot book size must be a multiple of 16 bytes")
    entries = book_path.stat().st_size // 16
    unique_keys = 0
    previous = None
    try:
        with book_path.open("rb") as handle:
            import struct
            while record := handle.read(16):
                key = struct.unpack(">Q", record[:8])[0]
                if key != previous:
                    unique_keys += 1
                    previous = key
    except OSError as exc:
        raise OpeningBookError(f"could not inspect opening book: {exc}") from exc
    return {"path": str(book_path), "bytes": book_path.stat().st_size,
            "entries": entries, "unique_position_keys": unique_keys}
