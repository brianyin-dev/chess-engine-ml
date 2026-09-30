"""Audit evaluator consistency and the positions actually evaluated during search."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median

import chess
import chess.engine
import numpy as np
import torch

from benchmarks.run import check_move, load_positions
from engine.evaluation import evaluate
from engine.search import search
from ml.evaluator import NeuralEvaluator
from ml.model import board_to_array, material_score, SCORE_SCALE


def quantiles(values):
    return {'n': len(values), 'mean': round(mean(values), 2),
            'median': round(median(values), 2),
            'p95': round(float(np.percentile(values, 95)), 2),
            'max': round(max(values), 2)} if values else {'n': 0}


def characteristics(boards):
    return {'positions': len(boards),
            'piece_count': quantiles([len(b.piece_map()) for b in boards]),
            'absolute_material_cp': quantiles([abs(material_score(b)) for b in boards]),
            'halfmove_clock': quantiles([b.halfmove_clock for b in boards]),
            'has_legal_capture_pct': round(100 * mean(bool(next(b.generate_legal_captures(), None))
                                                      for b in boards), 2),
            'in_check_pct': round(100 * mean(b.is_check() for b in boards), 2),
            'low_material_pct': round(100 * mean(len(b.piece_map()) <= 12 for b in boards), 2)}


def reference(engine, board):
    exact = None
    with engine.analysis(board, chess.engine.Limit(depth=10), game=object()) as analysis:
        for info in analysis:
            if 'score' in info and not info.get('lowerbound') and not info.get('upperbound'):
                exact = info['score'].white()
    if exact is None:
        raise ValueError('No exact Stockfish score')
    return {'cp': exact.score(), 'mate': exact.mate(),
            'score_cp': max(-1500, min(1500, exact.score(mate_score=1500)))}


class SampledEvaluator:
    cacheable_by_fen = True

    def __init__(self, evaluator, rng, sample=20):
        self.evaluator, self.rng, self.sample = evaluator, rng, sample
        self.calls, self.fens = 0, []

    def __call__(self, board):
        self.calls += 1
        if len(self.fens) < self.sample:
            self.fens.append(board.fen())
        else:
            index = self.rng.randrange(self.calls)
            if index < self.sample:
                self.fens[index] = board.fen()
        return self.evaluator(board)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', type=Path, required=True)
    parser.add_argument('--games', action='append', type=Path, required=True)
    parser.add_argument('--stockfish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nn-weight', type=float, default=1)
    parser.add_argument('--quiet-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output directory must be new')
    args.output.mkdir(parents=True)
    rng = random.Random(419)
    rows = [json.loads(l) for l in (args.data / 'test.jsonl').read_text().splitlines()]
    sampled = rng.sample(rows, min(len(rows), 512))
    boards = [chess.Board(r['fen']) for r in sampled]
    training = [chess.Board(json.loads(l)['fen'])
                for l in (args.data / 'train.jsonl').read_text().splitlines()]
    models = {p.stem: NeuralEvaluator(p, args.nn_weight, args.quiet_only) for p in args.checkpoint}
    evaluators = {'heuristic': evaluate, **models}
    # Use identical legal perturbations for every evaluator.
    material_rng = random.Random(431)
    material_probes = {}
    for piece_type in range(1, 6):
        probes = []
        for b in boards[:200]:
            candidates = [(color, sq) for color in chess.COLORS
                          for sq in b.pieces(piece_type, color)]
            if not candidates:
                continue
            color, sq = material_rng.choice(candidates)
            changed = b.copy(stack=False)
            changed.remove_piece_at(sq)
            if changed.is_valid() and not changed.is_game_over():
                probes.append((b, changed, 1 if color == chess.BLACK else -1))
        material_probes[piece_type] = probes
    report = {'seed': 419, 'material_seed': 431, 'leaf_seed': 443,
              'policy': 'Fixed depth-three search, reservoir sample of actual static '
              'evaluator calls at quiescence stand-pat nodes; these are not necessarily final quiet '
              'leaves. Sampling adds overhead, so no speed claims are made from these searches. '
              'Mirror tests swap colors and flip ranks. Material tests remove a piece '
              'from legal positions and check the color-relative gain. Stockfish labels use depth ten and ±1500cp clipping, matching '
              'training. Unsaturated results exclude mate and clipped labels.',
              'checkpoint_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in args.checkpoint},
              'games_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.games},
              'nn_weight': args.nn_weight, 'quiet_only': args.quiet_only,
              'data_manifest_sha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
              'consistency': {}, 'training_distribution': characteristics(training),
              'search_roots': [], 'leaf_accuracy': {}}
    for name, evaluator in evaluators.items():
        errors = [abs(evaluator(b) + evaluator(b.mirror())) for b in boards]
        symmetry = {**quantiles(errors), 'over_25_cp_pct': round(100 * mean(e > 25 for e in errors), 2)}
        symmetry['examples'] = [{'fen': boards[i].fen(), 'score_cp': evaluator(boards[i]),
                                 'mirrored_score_cp': evaluator(boards[i].mirror()),
                                 'error_cp': errors[i]}
                                for i in sorted(range(len(boards)), key=lambda i: errors[i], reverse=True)[:5]]
        material = {}
        for piece_type in range(1, 6):
            gains = [(evaluator(changed) - evaluator(b)) * sign
                     for b, changed, sign in material_probes[piece_type]]
            material[chess.piece_name(piece_type)] = {**quantiles(gains),
                'minimum_gain_cp': min(gains) if gains else None,
                'positive_gain_pct': round(100 * mean(g > 0 for g in gains), 2) if gains else None}
        parity = None
        if name in models:
            model = models[name]
            differences = []
            with torch.inference_mode():
                for b in boards:
                    features = torch.from_numpy(board_to_array(b, model.input_size))
                    base = evaluate(b) if model.target_mode == 'residual' else (
                           material_score(b) if model.target_mode == 'material' else 0)
                    use_nn = not (model.quiet_only and (b.is_check() or next(b.generate_legal_captures(), None)))
                    expected = round(base + model.model(features).item() * SCORE_SCALE * model.correction_weight) if use_nn else base
                    differences.append(abs(expected - model(b)))
            parity = {**quantiles(differences), 'different_integer_scores': sum(x != 0 for x in differences)}
        tactics = []
        for case in load_positions():
            if not any(k in case for k in ('expect', 'best', 'avoid')):
                continue
            b = chess.Board(case['fen'])
            result = search(b, depth=3, eval_fn=evaluator)
            tactics.append({'id': case['id'], 'move': result.move.uci(),
                           'passed': check_move(case, b, result.move), 'depth': result.depth})
        hanging = chess.Board('3r3k/8/8/3Q4/8/8/8/K7 b - - 0 1')
        safe = chess.Board('3r3k/8/8/8/4Q3/8/8/K7 b - - 0 1')
        hanging_scores = {'white_queen_hanging_cp': evaluator(hanging),
                          'white_queen_safe_cp': evaluator(safe),
                          'black_queen_hanging_cp': evaluator(hanging.mirror()),
                          'black_queen_safe_cp': evaluator(safe.mirror())}
        report['consistency'][name] = {'symmetry': symmetry, 'material': material,
                                      'training_inference_parity': parity,
                                      'tactical_search': tactics, 'hanging_static_example': hanging_scores}
        print(name, 'symmetry mean', symmetry['mean'], 'cp; tactics',
              sum(t['passed'] for t in tactics), '/', len(tactics), flush=True)
    # Reconstruct move history and sample three stages of each previously lost game.
    roots = []
    for path in args.games:
        for game in json.loads(path.read_text())['games'][:4]:
            for fraction in (.1, .4, .75):
                b = chess.Board(game['initial_fen'])
                for move in game['opening_moves']:
                    b.push_uci(move)
                index = int(len(game['moves']) * fraction)
                for move in game['moves'][:index]:
                    b.push_uci(move['uci'])
                if not b.is_game_over():
                    roots.append((b, game['opening'], index))
    active = next(iter(models.values()))
    report['search_sampling_model'] = next(iter(models))
    fens, seen = [], set()
    leaf_rng = random.Random(443)
    for index, (board, opening, ply) in enumerate(roots, 1):
        sampler = SampledEvaluator(active, leaf_rng)
        result = search(board, depth=3, eval_fn=sampler)
        report['search_roots'].append({'fen': board.fen(), 'opening': opening,
            'game_ply': ply, 'depth': result.depth, 'nodes': result.nodes,
            'evaluation_calls': sampler.calls, 'move': result.move.uci()})
        for fen in sampler.fens:
            key = fen.rsplit(' ', 1)[0]
            if key not in seen:
                seen.add(key)
                fens.append(fen)
        print(f'Sampled search root {index}/{len(roots)}', flush=True)
    leaves = [chess.Board(fen) for fen in fens]
    report['search_leaf_distribution'] = characteristics(leaves)
    labeled = []
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve())) as engine:
        engine.configure({'Threads': 1, 'Hash': 32})
        report['stockfish'] = engine.id
        report['stockfish_sha256'] = hashlib.sha256(args.stockfish.read_bytes()).hexdigest()
        for index, b in enumerate(leaves, 1):
            labeled.append({'fen': b.fen(), **reference(engine, b)})
            if index % 100 == 0:
                print(f'Labeled {index}/{len(leaves)} search leaves', flush=True)
        report['hanging_reference'] = {'hanging': reference(engine, hanging), 'safe': reference(engine, safe)}
    (args.output / 'leaves.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in labeled))
    for name, evaluator in evaluators.items():
        subsets = {'heldout_training_distribution': [(chess.Board(r['fen']), r['score_cp']) for r in sampled],
                   'search_calls': [(b, r['score_cp']) for b, r in zip(leaves, labeled)],
                   'quiet_unsaturated_search_calls': [(b, r['cp']) for b, r in zip(leaves, labeled)
                       if r['cp'] is not None and abs(r['cp']) < 1500 and
                       next(b.generate_legal_captures(), None) is None]}
        report['leaf_accuracy'][name] = {}
        for subset, positions in subsets.items():
            errors = [abs(evaluator(b) - label) for b, label in positions]
            clipped = [abs(max(-1500, min(1500, evaluator(b))) - label) for b, label in positions]
            report['leaf_accuracy'][name][subset] = {'absolute_error_cp': quantiles(errors),
                                                    'clipped_prediction_error_cp': quantiles(clipped)}
    report['search_label_saturated_pct'] = round(100 * mean(abs(r['score_cp']) == 1500 for r in labeled), 2)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Audit saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
