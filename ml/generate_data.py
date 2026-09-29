"""Generate fresh Stockfish self-play positions and White-perspective labels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import chess
import chess.engine
import chess.polyglot


def split_for_game(game_id: int) -> str:
    """Keep all positions from one game in the same split."""
    return ("train", "train", "train", "train", "train", "train", "train", "train", "val", "test")[game_id % 10]


def generate(engine, games: int, plies: int, sample_every: int, label_depth: int,
             play_ms: float, seed: int, book, book_plies: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    rows = {"train": [], "val": [], "test": []}
    seen = set()
    for game_id in range(games):
        board = chess.Board()
        game_token = object()
        for ply in range(plies):
            if board.is_game_over(claim_draw=False):
                break
            if ply >= 8 and ply % sample_every == 0:
                key = board.fen()
                if key not in seen:
                    seen.add(key)
                    info = engine.analyse(board, chess.engine.Limit(depth=label_depth),
                                          game=game_token, info=chess.engine.INFO_SCORE)
                    score = info["score"].white().score(mate_score=1500)
                    if score is not None:
                        rows[split_for_game(game_id)].append({
                            "game_id": game_id, "ply": ply, "fen": key,
                            "score_cp": max(-1500, min(1500, score)),
                        })
            if ply < book_plies:
                try:
                    move = book.weighted_choice(board, random=rng).move
                except IndexError:
                    move = None
            else:
                move = None
            if move is None:
                move = engine.play(board, chess.engine.Limit(time=play_ms / 1000),
                                   game=game_token).move
            if move is None:
                break
            board.push(move)
        if (game_id + 1) % 20 == 0:
            print(f"Generated {game_id + 1}/{games} games", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--opening-book", type=Path, required=True,
                        help="Polyglot book for varied, established opening moves")
    parser.add_argument("--book-plies", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="New directory for JSONL splits")
    parser.add_argument("--games", type=int, default=160)
    parser.add_argument("--plies", type=int, default=48)
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--label-depth", type=int, default=8)
    parser.add_argument("--play-ms", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.games < 10 or args.plies < 12 or args.sample_every < 1 or args.label_depth < 1 or args.play_ms <= 0 or args.book_plies < 1:
        parser.error("games >= 10, plies >= 12, sample-every >= 1, label-depth >= 1, play-ms > 0, book-plies >= 1 required")
    if not args.stockfish.is_file() or not args.opening_book.is_file() or args.output.exists():
        parser.error("Stockfish and opening-book files must exist; output directory must be new")
    args.output.mkdir(parents=True)
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve()), timeout=20)
    try:
        engine.configure({"Threads": 1, "Hash": 32})
        with chess.polyglot.open_reader(args.opening_book) as book:
            rows = generate(engine, args.games, args.plies, args.sample_every,
                            args.label_depth, args.play_ms, args.seed, book, args.book_plies)
        engine_id = engine.id
    finally:
        engine.quit()
    for split, records in rows.items():
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "local Stockfish self-play from weighted Polyglot opening moves",
        "stockfish": engine_id,
        "stockfish_sha256": hashlib.sha256(args.stockfish.read_bytes()).hexdigest(),
        "opening_book_sha256": hashlib.sha256(args.opening_book.read_bytes()).hexdigest(),
        "config": {"games": args.games, "plies": args.plies,
                   "sample_every": args.sample_every, "label_depth": args.label_depth,
                   "play_ms": args.play_ms, "seed": args.seed,
                   "book_plies": args.book_plies},
        "positions": {name: len(data) for name, data in rows.items()},
        "games_per_split": dict(Counter(split_for_game(i) for i in range(args.games))),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["positions"], indent=2))


if __name__ == "__main__":
    main()
