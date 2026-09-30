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


def conversion_loss(row):
    """Confirmed missed mate or loss of a substantial evaluated advantage."""
    return row['missed_forced_mate'] or (
        row['best_score']['cp'] is not None and row['best_score']['cp'] >= 150
        and row['played_score']['cp'] is not None and row['played_score']['cp'] <= 50
        and (row['cp_loss'] or 0) >= 100)


def report_checkpoint(report, requested):
    """Diagnose each source game using its actual model, verified by hash."""
    expected = report.get('checkpoint_sha256')
    if not expected or hashlib.sha256(requested.read_bytes()).hexdigest() == expected:
        return requested
    for path in (Path(__file__).resolve().parents[1] / 'ml/artifacts').glob('*.pt'):
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path
    raise ValueError('source game checkpoint cannot be resolved; refusing mismatched NN diagnostics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stockfish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nn-weight', type=float, default=1)
    parser.add_argument('--quiet-only', action='store_true')
    parser.add_argument('--include-draws', action='store_true', help='Review draws for missed mate or >=150cp advantage lost to <=50cp')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    evaluator = NeuralEvaluator(args.checkpoint, args.nn_weight, args.quiet_only)
    result = {'policy': 'First >=100cp loss or mate deterioration, screened at 100ms and '
              'confirmed at 400ms per root search. Longer NN probes diagnose time sensitivity, '
              'not a definitive cause. Scores are from the root mover perspective.',
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'nn_weight': args.nn_weight, 'quiet_only': args.quiet_only,
              'include_draws': args.include_draws,
              'models_by_report': {},
              'draw_policy': 'Missed forced mate or >=150cp best move versus <=50cp played move, confirmed at 400ms. Cp advantage is not proof of a forced win.',
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.report},
              'games': [], 'pairs': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve())) as engine:
        engine.configure({'Threads': 1, 'Hash': 32})
        result['stockfish'] = engine.id
        for source in args.report:
            recorded = json.loads(source.read_text())
            checkpoint = report_checkpoint(recorded, args.checkpoint)
            config = recorded.get('config', {})
            evaluator = NeuralEvaluator(checkpoint, config.get('nn_weight', args.nn_weight),
                                        config.get('quiet_only', args.quiet_only),
                                        incremental=config.get('incremental', False))
            result['models_by_report'][str(source)] = {'checkpoint': str(checkpoint),
                'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'nn_weight': evaluator.correction_weight, 'quiet_only': evaluator.quiet_only,
                'incremental': evaluator.incremental}
            for game in recorded['games']:
                drawn = game['current_result'] == 'draw'
                if game['current_result'] != 'loss' and not (args.include_draws and drawn):
                    continue
                board = chess.Board(game['initial_fen'])
                for move in game['opening_moves']:
                    board.push_uci(move)
                first = None
                screened = 0
                best_screened_cp = None
                for ply, item in enumerate(game['moves'], 1):
                    move = chess.Move.from_uci(item['uci'])
                    if item['engine'] == 'current':
                        screened += 1
                        row = review(engine, board, move, .1)
                        cp = row['best_score']['cp']
                        if cp is not None:
                            best_screened_cp = cp if best_screened_cp is None else max(cp, best_screened_cp)
                        qualifies = conversion_loss if drawn else costly
                        if qualifies(row):
                            row = review(engine, board, move, .4)
                            if qualifies(row):
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
                                equal_depth = {}
                                for name, evaluation in [('heuristic', NeuralEvaluator(checkpoint, 0, True, incremental=True)),
                                                         ('NN', evaluator)]:
                                    depth_probe = search(board.copy(stack=True), depth=3, eval_fn=evaluation)
                                    depth_review = review(engine, board, depth_probe.move, .4)
                                    equal_depth[name] = {'move': depth_probe.move.uci(), 'depth': depth_probe.depth,
                                                         'nodes': depth_probe.nodes, 'qnodes': depth_probe.qnodes,
                                                         'review': depth_review}
                                first = {'equal_depth': equal_depth, 'ply': ply, 'fen': board.fen(), 'played': move.uci(),
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
                                        'result': game['current_result'],
                                        'best_screened_advantage_cp': best_screened_cp,
                                        'first_costly_move': first})
                args.output.write_text(json.dumps(result, indent=2) + '\n')
                print(f"Reviewed {len(result['games'])} games: "
                      f"{first['played_san'] if first else 'no confirmed early mistake'}", flush=True)
    result['summary'] = {'games': len(result['games']),
        'draws_reviewed': sum(g['result'] == 'draw' for g in result['games']),
        'draws_with_confirmed_conversion_loss': sum(g['result'] == 'draw' and g['first_costly_move'] is not None for g in result['games']),
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
