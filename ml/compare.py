"""Paired neural-versus-heuristic games under equal per-move budgets."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import chess

from benchmarks.match import UciOpponent, choose_move, opening_board, play_game, summarize
from engine.search import search
from ml.evaluator import NeuralEvaluator

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openings", type=Path, default=ROOT / "benchmarks/openings-live-2026.json")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--time-ms", type=float, default=250)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--stockfish", type=Path, help="Local UCI opponent instead of the heuristic")
    parser.add_argument("--stockfish-elo", type=int, help="Stockfish limited-strength setting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    openings = json.loads(args.openings.read_text())
    for opening in openings:
        opening_board(opening)
    if not 1 <= args.pairs <= len(openings) or args.time_ms <= 0 or args.max_plies < 1:
        parser.error("invalid pairs, time-ms, or max-plies")
    if args.stockfish_elo is not None and args.stockfish is None:
        parser.error("stockfish-elo requires --stockfish")
    if args.output.exists():
        parser.error("output directory already exists")
    evaluator = NeuralEvaluator(args.checkpoint)
    uci = UciOpponent(args.stockfish, args.stockfish_elo) if args.stockfish else None
    opponent = (f"stockfish-elo-{args.stockfish_elo}" if args.stockfish_elo is not None
                else "stockfish" if uci else "classical")

    def select(name, board, time_limit, depth_cap):
        if name == "current":
            result = search(board, depth=depth_cap, eval_fn=evaluator, time_limit=time_limit)
            stats = asdict(result)
            stats.pop("move")
            stats["budget_seconds"] = time_limit
            stats["overrun_seconds"] = max(0, result.elapsed - time_limit)
            return result.move, stats
        # The NN search retains the app's depth-eight cap. A UCI opponent must
        # be free to use its full clock; capping Stockfish at eight can make a
        # nominal 250 ms match consume only a few milliseconds per move.
        return (uci(name, board, time_limit, 64) if uci else
                choose_move("current", board, time_limit, depth_cap))

    args.output.mkdir(parents=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "policy": "NN is current, selected local engine is opponent. Equal per-move "
                  "cooperative budgets, no opening book, sequential paired colors. "
                  "Ply-limit games are unfinished. No Elo inference.",
        "config": {"pairs": args.pairs, "time_ms": args.time_ms,
                   "max_plies": args.max_plies, "depth_cap": 8,
                   "opponent_depth_cap": 64 if uci else 8,
                   "target_mode": evaluator.target_mode},
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "openings_sha256": hashlib.sha256(args.openings.read_bytes()).hexdigest(),
        "games": [],
    }
    if uci:
        report["opponent"] = {"id": uci.engine.id, "options": uci.options,
                              "sha256": hashlib.sha256(args.stockfish.read_bytes()).hexdigest()}
    pgns = []
    try:
        for pair_id, opening in enumerate(openings[:args.pairs], 1):
            colors = (chess.WHITE, chess.BLACK) if pair_id % 2 else (chess.BLACK, chess.WHITE)
            for color in colors:
                if uci:
                    uci.new_game()
                print(f"Game {len(report['games']) + 1}/{args.pairs * 2}: "
                      f"{opening['name']}, NN as {'White' if color else 'Black'}", flush=True)
                game, pgn = play_game(opening, color, args.time_ms / 1000,
                                      args.max_plies, depth_cap=8, selector=select,
                                      pair_id=pair_id, opponent=opponent)
                report["games"].append(game)
                pgns.append(pgn)
                report["summary"] = summarize(report["games"])
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                (args.output / "games.pgn").write_text("".join(pgns))
                print(f"  {game['result']} ({game['reason']}, {game['played_plies']} plies)", flush=True)
    finally:
        if uci:
            uci.close()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
