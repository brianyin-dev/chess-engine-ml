"""Find and confirm the first costly neural-engine move in each lost game."""
import argparse
import hashlib
import json
from pathlib import Path

import chess
import chess.engine

from benchmarks.analyze import review
from engine.search import search
from ml.evaluator import NeuralEvaluator


def costly(row):
    return row['allows_mate'] or row['missed_forced_mate'] or (row['cp_loss'] or 0) >= 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stockfish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nn-weight', type=float, default=1)
    parser.add_argument('--quiet-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    evaluator = NeuralEvaluator(args.checkpoint, args.nn_weight, args.quiet_only)
    result = {'policy': 'First >=100cp loss or mate deterioration, screened at 100ms and '
              'confirmed at 400ms per root search. Longer NN probes diagnose time sensitivity, '
              'not a definitive cause. Scores are from the root mover perspective.',
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'nn_weight': args.nn_weight, 'quiet_only': args.quiet_only,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.report},
              'games': [], 'pairs': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve())) as engine:
        engine.configure({'Threads': 1, 'Hash': 32})
        result['stockfish'] = engine.id
        for source in args.report:
            for game in json.loads(source.read_text())['games']:
                if game['current_result'] != 'loss':
                    continue
                board = chess.Board(game['initial_fen'])
                for move in game['opening_moves']:
                    board.push_uci(move)
                first = None
                screened = 0
                for ply, item in enumerate(game['moves'], 1):
                    move = chess.Move.from_uci(item['uci'])
                    if item['engine'] == 'current':
                        screened += 1
                        row = review(engine, board, move, .1)
                        if costly(row):
                            row = review(engine, board, move, .4)
                            if costly(row):
                                alternative = chess.Move.from_uci(row['best_move'])
                                good, bad = board.copy(), board.copy()
                                good.push(alternative)
                                bad.push(move)
                                result['pairs'].append({'good_fen': good.fen(), 'bad_fen': bad.fen(),
                                    'sign': 1 if board.turn else -1, 'source': str(source),
                                    'game': len(result['games']), 'cp_loss': row['cp_loss']})
                                probe = search(board.copy(stack=True), depth=8, time_limit=1,
                                               eval_fn=evaluator)
                                probe_review = review(engine, board, probe.move, .4)
                                def mover_score(position):
                                    return evaluator(position) * (1 if board.turn else -1)
                                first = {'ply': ply, 'fen': board.fen(), 'played': move.uci(),
                                         'played_san': board.san(move), **row,
                                         'nn_static_good_cp': mover_score(good),
                                         'nn_static_bad_cp': mover_score(bad),
                                         'original_depth': item['depth'], 'probe_depth': probe.depth,
                                         'probe_move': probe.move.uci(), 'probe_review': probe_review,
                                         'extra_time_recovered': not costly(probe_review)}
                                break
                    board.push(move)
                result['games'].append({'opening': game['opening'], 'color': game['current_color'],
                                        'source': str(source), 'screened_moves': screened,
                                        'first_costly_move': first})
                args.output.write_text(json.dumps(result, indent=2) + '\n')
                print(f"Reviewed {len(result['games'])} games: "
                      f"{first['played_san'] if first else 'no confirmed early mistake'}", flush=True)
    result['summary'] = {'games': len(result['games']),
        'confirmed': sum(g['first_costly_move'] is not None for g in result['games']),
        'extra_time_recovered': sum(bool(g['first_costly_move'] and
            g['first_costly_move']['extra_time_recovered']) for g in result['games']),
        'wrong_static_ranking': sum(bool(g['first_costly_move'] and
            g['first_costly_move']['nn_static_bad_cp'] >= g['first_costly_move']['nn_static_good_cp'])
            for g in result['games'])}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['summary'])


if __name__ == '__main__':
    main()
