"""Add Stockfish-labeled legal-move deviations to an existing game-split dataset."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import chess
import chess.engine

from ml.dataset import load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.33)
    parser.add_argument("--label-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=229)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or args.label_depth < 1:
        parser.error("fraction must be in (0, 1] and label-depth positive")
    if not args.stockfish.is_file() or args.output.exists():
        parser.error("Stockfish must exist and output directory must be new")
    rows = {split: load_rows(args.data / f"{split}.jsonl")
            for split in ("train", "val", "test")}
    seen = {row["fen"] for records in rows.values() for row in records}
    rng = random.Random(args.seed)
    added = {split: 0 for split in rows}
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve()), timeout=20)
    try:
        engine.configure({"Threads": 1, "Hash": 32})
        for split, records in rows.items():
            original = list(records)
            for index, parent in enumerate(original, 1):
                if rng.random() >= args.fraction:
                    continue
                board = chess.Board(parent["fen"])
                legal = list(board.legal_moves)
                if not legal:
                    continue
                board.push(rng.choice(legal))
                fen = board.fen()
                if fen in seen:
                    continue
                seen.add(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=args.label_depth),
                                      info=chess.engine.INFO_SCORE)
                score = info["score"].white().score(mate_score=1500)
                if score is None:
                    continue
                records.append({"game_id": parent["game_id"], "ply": parent["ply"] + 1,
                                "fen": fen, "score_cp": max(-1500, min(1500, score)),
                                "source": "random_legal_successor"})
                added[split] += 1
                if split == "train" and added[split] % 500 == 0:
                    print(f"Scanned {index}/{len(original)} training positions", flush=True)
    finally:
        engine.quit()
    args.output.mkdir(parents=True)
    for split, records in rows.items():
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "existing book-started self-play positions plus random legal successors",
        "parent_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
        "stockfish_sha256": hashlib.sha256(args.stockfish.read_bytes()).hexdigest(),
        "config": {"fraction": args.fraction, "label_depth": args.label_depth,
                   "seed": args.seed},
        "added": added,
        "positions": {split: len(records) for split, records in rows.items()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["positions"], indent=2))


if __name__ == "__main__":
    main()
