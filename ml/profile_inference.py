"""Profile standalone evaluator calls on a fixed sampled set of positions."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import chess

from engine.evaluation import evaluate
from ml.evaluator import NeuralEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True, help="JSONL FEN positions")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--sample", type=int, default=250)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--nn-weight', type=float, default=1)
    parser.add_argument('--quiet-only', action='store_true')
    args = parser.parse_args()
    if args.sample < 1 or args.rounds < 1 or args.output.exists():
        parser.error("sample and rounds must be positive; output must be new")
    records = [json.loads(line) for line in args.positions.read_text().splitlines() if line]
    if len(records) < args.sample:
        parser.error("fewer positions than requested sample")
    indices = [i * len(records) // args.sample for i in range(args.sample)]
    boards = [chess.Board(records[i]["fen"]) for i in indices]
    evaluators = {"heuristic": evaluate}
    for path in args.checkpoint:
        evaluators[path.stem] = NeuralEvaluator(path, args.nn_weight, args.quiet_only)
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "positions_sha256": hashlib.sha256(args.positions.read_bytes()).hexdigest(),
              "sample": args.sample, "rounds": args.rounds,
              "nn_weight": args.nn_weight, "quiet_only": args.quiet_only, "evaluators": {}}
    for name, evaluator in evaluators.items():
        for board in boards:
            evaluator(board)
        times = []
        for _ in range(args.rounds):
            start = perf_counter()
            for board in boards:
                evaluator(board)
            times.append((perf_counter() - start) * 1000 / len(boards))
        result["evaluators"][name] = {"median_ms_per_position": round(median(times), 5),
                                      "rounds_ms_per_position": [round(x, 5) for x in times]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["evaluators"], indent=2))


if __name__ == "__main__":
    main()
