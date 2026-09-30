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
from benchmarks.nn_baseline_v10.search import search as frozen_search
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
    parser.add_argument('--opponent-checkpoint', type=Path, help='Neural baseline instead of the heuristic')
    parser.add_argument('--opponent-nn-weight', type=float, default=1.)
    parser.add_argument('--opponent-quiet-only', action='store_true')
    parser.add_argument('--opponent-incremental', action='store_true')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--nodes', type=int, help='Equal visited-node budgets, including quiescence; no clock limit')
    parser.add_argument('--nn-weight', type=float, default=1.0)
    parser.add_argument('--quiet-only', action='store_true', help='Apply correction only outside check with no legal capture')
    parser.add_argument('--incremental', action='store_true', help='Reuse unchanged neural and handcrafted feature blocks')
    parser.add_argument('--reference-inference', action='store_true', help='Disable the NN search-leaf fast path')
    parser.add_argument('--frozen-heuristic', action='store_true', help='Use the heuristic search frozen at 9e23b7c as local opponent')
    args = parser.parse_args()
    if args.nodes is not None and (args.nodes < 1 or args.stockfish):
        parser.error('positive node budget requires a local opponent')
    openings = json.loads(args.openings.read_text())
    for opening in openings:
        opening_board(opening)
    if not 1 <= args.pairs <= len(openings) or args.time_ms <= 0 or args.max_plies < 1:
        parser.error("invalid pairs, time-ms, or max-plies")
    if args.stockfish_elo is not None and args.stockfish is None:
        parser.error("stockfish-elo requires --stockfish")
    if args.stockfish and args.opponent_checkpoint:
        parser.error('choose either Stockfish or a neural baseline')
    if args.frozen_heuristic and (args.stockfish or args.opponent_checkpoint):
        parser.error('frozen-heuristic requires a local heuristic opponent')
    if (args.opponent_nn_weight != 1 or args.opponent_quiet_only or args.opponent_incremental) and not args.opponent_checkpoint:
        parser.error('opponent blend/gate requires opponent-checkpoint')
    if args.output.exists():
        parser.error("output directory already exists")
    evaluator = NeuralEvaluator(args.checkpoint, args.nn_weight, args.quiet_only,
                                optimized=not args.reference_inference, incremental=args.incremental)
    if evaluator.diagnostic_only:
        parser.error("memorization-only checkpoints are excluded from match candidates")
    uci = UciOpponent(args.stockfish, args.stockfish_elo) if args.stockfish else None
    neural_baseline = NeuralEvaluator(args.opponent_checkpoint, args.opponent_nn_weight,
                                     args.opponent_quiet_only, incremental=args.opponent_incremental) if args.opponent_checkpoint else None
    if neural_baseline and neural_baseline.diagnostic_only:
        parser.error('memorization-only checkpoints are excluded from opponents')
    opponent = (f"stockfish-elo-{args.stockfish_elo}" if args.stockfish_elo is not None
                else "stockfish" if uci else 'neural-baseline' if neural_baseline else "classical")

    def select(name, board, time_limit, depth_cap):
        selected = (evaluator if args.nn_weight or args.incremental else None) if name == 'current' else neural_baseline
        if selected is not None or args.nodes is not None or (args.frozen_heuristic and name != 'current' and not uci):
            search_fn = frozen_search if args.frozen_heuristic and name != 'current' else search
            result = search_fn(board, depth=depth_cap, eval_fn=selected,
                            time_limit=None if args.nodes else time_limit,
                            node_limit=args.nodes)
            stats = asdict(result)
            stats.pop("move")
            stats["budget_seconds"] = None if args.nodes else time_limit
            stats["budget_nodes"] = args.nodes
            stats["overrun_seconds"] = 0 if args.nodes else max(0, result.elapsed - time_limit)
            return result.move, stats
        # The NN search retains the app's depth-eight cap. A UCI opponent must
        # be free to use its full clock; capping Stockfish at eight can make a
        # nominal 250 ms match consume only a few milliseconds per move.
        return (uci(name, board, time_limit, 64) if uci else
                choose_move("current", board, time_limit, depth_cap))

    args.output.mkdir(parents=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "policy": ("Candidate is current; local opponent. Equal visited-node budgets including quiescence; "
                   "fallback static evaluations are reported separately. No clock limit. "
                   if args.nodes else "Candidate is current; selected opponent. Equal cooperative per-move clock budgets. ") +
                  "No opening book, sequential paired colors. Ply-limit games are unfinished. No Elo inference.",
        "config": {"pairs": args.pairs, "time_ms": args.time_ms,
                   "max_plies": args.max_plies, "depth_cap": 8, "node_limit": args.nodes,
                   "opponent_depth_cap": 64 if uci else 8,
                   "nn_weight": args.nn_weight, "quiet_only": args.quiet_only,
                   "reference_inference": args.reference_inference,
                   'incremental': args.incremental,
                   "frozen_heuristic": args.frozen_heuristic,
                   "target_mode": evaluator.target_mode, 'input_size': evaluator.input_size,
                   'hidden_sizes': list(evaluator.hidden_sizes),
                   'correction_limit_cp': evaluator.correction_limit_cp},
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "openings_sha256": hashlib.sha256(args.openings.read_bytes()).hexdigest(),
        "engine_sources_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT / 'engine').glob('*.py'))},
        'evaluation_sources_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (ROOT/'ml/evaluator.py', ROOT/'ml/model.py', ROOT/'ml/incremental.py')},
        "candidate": "weighted-neural" if args.nn_weight else ('incremental-heuristic' if args.incremental else 'heuristic'),
        "games": [],
    }
    if uci:
        report["opponent"] = {"id": uci.engine.id, "options": uci.options,
                              "sha256": hashlib.sha256(args.stockfish.read_bytes()).hexdigest()}
    elif neural_baseline:
        report['opponent'] = {'checkpoint_sha256': hashlib.sha256(args.opponent_checkpoint.read_bytes()).hexdigest(),
                              'correction_weight': args.opponent_nn_weight, 'quiet_only': args.opponent_quiet_only,
                              'incremental': args.opponent_incremental,
                              'target_mode': neural_baseline.target_mode,
                              'input_size': neural_baseline.input_size}
        report['opponent']['hidden_sizes'] = list(neural_baseline.hidden_sizes)
    elif args.frozen_heuristic:
        frozen_path = ROOT / 'benchmarks/nn_baseline_v10/search.py'
        report['opponent'] = {'search_frozen_at': '9e23b7c',
                              'search_sha256': hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                              'evaluation_sha256': hashlib.sha256((ROOT / 'benchmarks/nn_baseline_v10/evaluation.py').read_bytes()).hexdigest()}
    pgns = []
    try:
        for pair_id, opening in enumerate(openings[:args.pairs], 1):
            colors = (chess.WHITE, chess.BLACK) if pair_id % 2 else (chess.BLACK, chess.WHITE)
            for color in colors:
                if uci:
                    uci.new_game()
                print(f"Game {len(report['games']) + 1}/{args.pairs * 2}: "
                      f"{opening['name']}, candidate as {'White' if color else 'Black'}", flush=True)
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
    from benchmarks.cpu_lock import exclusive_cpu
    with exclusive_cpu('neural comparison'):
        main()
