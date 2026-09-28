"""Review current-engine moves with full-strength Stockfish; save reproducible evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import chess
import chess.engine

from engine.search import search


def score_data(score, color):
    pov = score.pov(color)
    return {"cp": pov.score(), "mate": pov.mate()}


def loss_data(best, played):
    cp_loss = max(0, best['cp'] - played['cp']) if best['cp'] is not None and played['cp'] is not None else None
    allows_mate = (played['mate'] is not None and played['mate'] < 0
                   and (best['mate'] is None or best['mate'] > 0))
    missed_mate = best['mate'] is not None and best['mate'] > 0 and played['mate'] is None
    return {"cp_loss": cp_loss, "allows_mate": allows_mate, "missed_forced_mate": missed_mate}


def san_line(board, moves):
    copy = board.copy()
    line = []
    for move in moves[:12]:
        line.append(copy.san(move))
        copy.push(move)
    return line


def reference_search(engine, board, seconds, root_moves=None):
    """Keep the last exact-score update, not merged/transient aspiration bounds.

    SimpleEngine.analyse() merges info updates, including stale bound flags and
    potentially an unfinished iteration's bound score. Consume raw updates so a
    stopped search cannot replace a completed score with a lower/upper bound.
    """
    exact = None
    with engine.analysis(board, chess.engine.Limit(time=seconds), game=object(),
                         root_moves=root_moves) as analysis:
        for info in analysis:
            if ('score' in info and info.get('pv') and
                    not info.get('lowerbound') and not info.get('upperbound')):
                exact = dict(info)
    if exact is None:
        raise ValueError('Reference search produced no exact scored principal variation; increase its budget')
    return exact


def review(engine, board, played, seconds):
    # A fresh game marker clears state between root searches. Both candidate
    # scores are from the mover's perspective at the same root, not mixed plies.
    best = reference_search(engine, board, seconds)
    best_move = best['pv'][0]
    actual = best if played == best_move else reference_search(engine, board, seconds, root_moves=[played])
    best_score, actual_score = score_data(best['score'], board.turn), score_data(actual['score'], board.turn)
    return {"best_move": best_move.uci(), "best_san": board.san(best_move),
            "best_score": best_score, "played_score": actual_score,
            "best_pv": san_line(board, best.get('pv', [])),
            "played_pv": san_line(board, actual.get('pv', [])),
            "best_depth": best.get('depth'), "played_depth": actual.get('depth'),
            "analysis_seconds_per_search": seconds, **loss_data(best_score, actual_score)}


def significance(row):
    best = row['best_score']
    # Prioritize damage in positions not already clearly lost; mate scores are
    # handled explicitly rather than fabricated as giant centipawn losses.
    competitive = best['cp'] is not None and best['cp'] >= -300 or best['mate'] is not None and best['mate'] > 0
    return (int(competitive), int(row['allows_mate']), row['cp_loss'] or 0)


def replay(row):
    board = chess.Board(row['initial_fen'])
    for uci in row['history']:
        board.push_uci(uci)
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--stockfish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-ms', type=float, default=100)
    parser.add_argument('--confirm-count', type=int, default=6)
    parser.add_argument('--confirm-ms', type=float, default=500)
    parser.add_argument('--probe-ms', type=float, default=1500)
    args = parser.parse_args()
    if any(not 0 < v < float('inf') for v in [args.time_ms, args.confirm_ms, args.probe_ms]) or args.confirm_count < 0:
        parser.error('positive finite budgets and nonnegative confirm-count required')
    if args.output.exists():
        parser.error('output already exists; choose a new file')
    source_bytes = args.report.read_bytes()
    source = json.loads(source_bytes)
    if source['status'] == 'running':
        parser.error('wait for the tournament to finish before reviewing it')
    document = {"created_at": datetime.now(timezone.utc).isoformat(),
                "source_report": str(args.report.resolve()), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "reviewer_binary_sha256": hashlib.sha256(args.stockfish.read_bytes()).hexdigest(),
                "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "probe_source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__).resolve().parents[1] / 'engine/search.py',
                                 Path(__file__).resolve().parents[1] / 'engine/evaluation.py']},
                "python_chess": chess.__version__, "config": {
                    "time_ms": args.time_ms, "confirm_ms": args.confirm_ms,
                    "probe_ms": args.probe_ms, "confirm_count": args.confirm_count},
                "policy": "Root-restricted reference searches; heuristic estimates, not proofs. "
                          "Largest competitive-position losses are confirmed with more time. "
                          "Longer current-engine probes diagnose search sensitivity, not causation.",
                "moves": [], "confirmed": [], "status": "running"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(document, indent=2) + '\n')
        temp.replace(args.output)
    save()
    try:
        with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve()), timeout=10) as engine:
            engine.configure({'Threads':1, 'Hash':32})
            document['reviewer'] = engine.id
            for game_index, game in enumerate(source['games'], 1):
                board = chess.Board(game['initial_fen'])
                history = list(game['opening_moves'])
                for move in history:
                    board.push_uci(move)
                for ply, move in enumerate(game['moves'], 1):
                    played = chess.Move.from_uci(move['uci'])
                    if move['engine'] == 'current':
                        row = {"game": game_index, "opening": game['opening'], "current_color": game['current_color'],
                               "game_result": game['current_result'], "played_ply": ply,
                               "move_number": board.fullmove_number, "fen": board.fen(),
                               "initial_fen": game['initial_fen'], "history": list(history),
                               "played": played.uci(), "san": board.san(played),
                               "engine_depth": move['depth'], "engine_score": move.get('score'),
                               "engine_elapsed": move['elapsed'], "engine_nodes": move.get('nodes'),
                               "engine_qnodes": move.get('qnodes'),
                               **review(engine, board, played, args.time_ms / 1000)}
                        document['moves'].append(row)
                    board.push(played)
                    history.append(played.uci())
                save()
                print(f"Reviewed game {game_index}/{len(source['games'])}", flush=True)
            selected, games_seen = [], set()
            for row in sorted(document['moves'], key=significance, reverse=True):
                if row['game'] not in games_seen and (row['allows_mate'] or (row['cp_loss'] or 0) >= 100):
                    selected.append(row)
                    games_seen.add(row['game'])
                if len(selected) >= args.confirm_count:
                    break
            for row in selected[:args.confirm_count]:
                board = replay(row)
                played = chess.Move.from_uci(row['played'])
                confirmed = {**row, **review(engine, board, played, args.confirm_ms / 1000)}
                probe = search(board, depth=64, time_limit=args.probe_ms / 1000)
                confirmed['longer_search'] = {"move": probe.move.uci(), "san": board.san(probe.move),
                    "depth": probe.depth, "elapsed": probe.elapsed, "nodes": probe.nodes,
                    **review(engine, board, probe.move, args.confirm_ms / 1000)}
                document['confirmed'].append(confirmed)
                save()
                print(f"Confirmed game {row['game']}, {row['move_number']} {row['san']}", flush=True)
    except KeyboardInterrupt:
        document['status'] = 'interrupted'
        save()
        raise SystemExit(130)
    document['status'] = 'completed'
    document['summary'] = {
        "reviewed_moves": len(document['moves']),
        "estimated_losses_100cp_or_more": sum((r['cp_loss'] or 0) >= 100 for r in document['moves']),
        "allows_mate": sum(r['allows_mate'] for r in document['moves']),
        "depth_zero_moves": sum(r['engine_depth'] == 0 for r in document['moves']),
        "confirmed_positions": len(document['confirmed']),
    }
    save()
    print(json.dumps(document['summary'], indent=2))


if __name__ == '__main__':
    main()
