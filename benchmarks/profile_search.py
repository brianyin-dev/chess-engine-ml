"""Deterministic fixed-depth search timings for before/after profiling."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import chess
from engine.search import search

POSITIONS = Path(__file__).with_name("positions.json")
IDS = {"starting-position", "middlegame", "poisoned-pawn"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.depth <= 6 or args.output.exists():
        parser.error("choose repeats >= 1, depth 1–6, and a new output file")
    rows = []
    for fixture in json.loads(POSITIONS.read_text()):
        if fixture["id"] not in IDS:
            continue
        board = chess.Board(fixture["fen"])
        trials = []
        for _ in range(args.repeats):
            start = perf_counter()
            result = search(board, depth=args.depth)
            trials.append({"elapsed": perf_counter() - start, "nodes": result.nodes,
                           "move": result.move.uci(), "score": result.score})
        assert len({(t["move"], t["score"], t["nodes"]) for t in trials}) == 1
        rows.append({"id": fixture["id"], "median_seconds": median(t["elapsed"] for t in trials),
                     "nodes": trials[0]["nodes"], "move": trials[0]["move"],
                     "score": trials[0]["score"], "trials": trials})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"depth": args.depth, "repeats": args.repeats,
                                       "positions": rows}, indent=2) + "\n")
    for row in rows:
        print(f"{row['id']}: {row['median_seconds']:.4f}s, {row['nodes']} nodes, {row['move']}")


if __name__ == "__main__":
    main()
