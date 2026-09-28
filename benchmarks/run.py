"""Run a small regression/performance suite, not an Elo estimator."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import chess
from benchmarks.legacy.search import best_move as legacy_move
from engine.search import search

ROOT = Path(__file__).resolve().parents[1]


def load_positions():
    return json.loads((ROOT / "benchmarks/positions.json").read_text())


def check_move(case, board, move):
    if move not in board.legal_moves:
        return False
    if case.get("expect") == "mate":
        after = board.copy()
        after.push(move)
        return after.is_checkmate()
    if "best" in case:
        return move.uci() in case["best"]
    if "avoid" in case:
        return move.uci() not in case["avoid"]
    return None  # A legal performance sample is not a solved tactical problem.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["current", "legacy", "both"], default="both")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time-ms", type=float, help="Current engine only; legacy has no deadline support")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.depth <= 64:
        parser.error("depth must be between 1 and 64")
    if args.time_ms is not None and (args.engine != "current" or not 0 < args.time_ms < float("inf")):
        parser.error("a positive finite --time-ms requires --engine current")
    engines = ["legacy", "current"] if args.engine == "both" else [args.engine]
    rows = []
    for engine in engines:
        for case in load_positions():
            board = chess.Board(case["fen"])
            if not board.is_valid():
                raise ValueError(f"Invalid fixture: {case['id']}")
            start = perf_counter()
            if engine == "current":
                result = search(board, depth=args.depth,
                                time_limit=None if args.time_ms is None else args.time_ms / 1000)
                move = result.move
                stats = asdict(result)
                del stats["move"]
            else:
                move = legacy_move(board, depth=args.depth)
                stats = {"elapsed": perf_counter() - start, "depth": args.depth,
                         "nodes": None, "qnodes": None, "tt_hits": None, "score": None}
            if move not in board.legal_moves:
                raise AssertionError(f"{engine}: illegal move for {case['id']}")
            passed = check_move(case, board, move)
            rows.append({"engine": engine, "id": case["id"], "fen": case["fen"],
                         "move": move.uci(), "san": board.san(move), "passed": passed, **stats})
            status = "sample" if passed is None else "PASS" if passed else "FAIL"
            print(f"{engine:7} {case['id']:18} {board.san(move):8} {status:6} {stats['elapsed']:.3f}s")
    sources = ["engine/search.py", "engine/evaluation.py", "benchmarks/legacy/search.py",
               "benchmarks/legacy/evaluation.py", "benchmarks/positions.json", "benchmarks/run.py"]
    report = {
        "python": sys.version, "platform": platform.platform(), "python_chess": chess.__version__,
        "depth_limit": args.depth, "time_ms": args.time_ms,
        "comparison": "Fixed main-search depth; quiescence adds work. Not equal-time or Elo evidence.",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "results": rows,
    }
    for engine in engines:
        tested = [r for r in rows if r["engine"] == engine and r["passed"] is not None]
        print(f"{engine}: {sum(r['passed'] for r in tested)}/{len(tested)} tactical checks")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    # Expected legacy failures should not make a current-engine regression run fail.
    if any(r["passed"] is False and r["engine"] == "current" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
