"""Budgeted Stockfish distillation: search leaves, hard examples, endgames, rankings."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import chess
import chess.engine
from engine.evaluation import evaluate
from engine.search import search
from ml.audit import SampledEvaluator
from ml.dataset import load_rows
from ml.evaluator import NeuralEvaluator
from ml.generate_search_data import key


def score(info):
    return max(-1500, min(1500, info['score'].white().score(mate_score=1500)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stockfish', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-positions', type=int, default=10000)
    p.add_argument('--holdout-positions', type=int, default=1000)
    args = p.parse_args()
    if args.output.exists() or min(args.train_positions, args.holdout_positions) < 1:
        p.error('use new output and positive position counts')
    args.output.mkdir(parents=True)
    pairdir = args.output / 'pairs'
    pairdir.mkdir()
    rng = random.Random(719)
    started = time.monotonic()
    base = {s: load_rows(args.data / f'{s}.jsonl') for s in ('train', 'val', 'test')}
    owners = {key(r['fen']): s for s, rows in base.items() for r in rows}
    if len(owners) != sum(len(rows) for rows in base.values()):
        raise ValueError('source contains canonical aliases; use cleaned v6 data')
    model = NeuralEvaluator(args.checkpoint)
    label_cache = {}
    counts = {}
    total_budget = 0
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve())) as sf:
        sf.configure({'Threads': 1, 'Hash': 64})
        for split, rows in base.items():
            target = args.train_positions if split == 'train' else args.holdout_positions
            added, pairs, roots, refined = [], [], 0, 0
            sources = {}
            # Keep source-game splits intact; prioritize known disagreement and endgames.
            eligible = [r for r in rows if not chess.Board(r['fen']).is_game_over()]
            hard = sorted(eligible, key=lambda r: abs(r['score_cp'] - evaluate(chess.Board(r['fen']))), reverse=True)[:max(1, len(eligible)//3)]
            endings = [r for r in eligible if len(chess.Board(r['fen']).piece_map()) <= 12]
            def label(board, close):
                nonlocal refined, total_budget
                fen = board.fen()
                if fen in label_cache:
                    return label_cache[fen]
                first = sf.analyse(board, chess.engine.Limit(nodes=500), game=object())
                second = sf.analyse(board, chess.engine.Limit(nodes=1500))
                total_budget += 2000
                initial, value = score(first), score(second)
                reason = ('unstable' if abs(initial - value) > 100 else
                          'close_candidates' if close else
                          'student_disagreement' if abs(model(board) - value) > 200 else None)
                upgraded = reason is not None and refined < target // 4
                if upgraded:
                    value = score(sf.analyse(board, chess.engine.Limit(nodes=6000)))
                    total_budget += 6000
                    refined += 1
                result = {'score_cp': value, 'label_budget_nodes': 8000 if upgraded else 2000,
                          'refinement_reason': reason if upgraded else None,
                          'screen_score_cp': initial}
                label_cache[fen] = result
                return result
            def add(board, root, source, close=False):
                if len(added) >= target or board.is_game_over() or not board.is_valid():
                    return False
                canonical = key(board.fen())
                if canonical in owners:
                    return False
                owners[canonical] = split
                added.append({'fen': board.fen(), 'game_id': root['game_id'], 'ply': root['ply'],
                              'source': source, 'root_fen': root['fen'], **label(board, close)})
                sources[source] = sources.get(source, 0) + 1
                return True
            while len(added) < target:
                roots += 1
                if roots > target * 3:
                    raise RuntimeError('candidate pool exhausted before requested quota')
                pool = endings if roots % 4 == 0 and endings else hard if roots % 4 == 1 else eligible
                root = rng.choice(pool)
                board = chess.Board(root['fen'])
                sampler = SampledEvaluator(model, rng, 16)
                search(board, depth=8, node_limit=1500, eval_fn=sampler)
                for fen in sampler.fens:
                    leaf = chess.Board(fen)
                    # Quiet calls have evaluation targets closest to our search contract.
                    source = ('endgame_search' if len(leaf.piece_map()) <= 12 else
                              'quiet_search' if not next(leaf.generate_legal_captures(), None) else 'search_stand_pat')
                    add(leaf, root, source)
                if len(added) >= target:
                    break
                infos = sf.analyse(board, chess.engine.Limit(nodes=2000), multipv=2, game=object())
                total_budget += 2000
                if len(infos) >= 2 and infos[0].get('pv') and infos[1].get('pv'):
                    good, bad = board.copy(), board.copy()
                    good.push(infos[0]['pv'][0]); bad.push(infos[1]['pv'][0])
                    delta = (score(infos[0]) - score(infos[1])) * (1 if board.turn else -1)
                    close = abs(delta) < 50
                    add(good, root, 'teacher_candidate', close)
                    add(bad, root, 'teacher_candidate', close)
                    if all(owners.get(key(b.fen())) == split for b in (good, bad)) and not any(b.is_game_over() for b in (good,bad)):
                        good_label, bad_label = label(good, close), label(bad, close)
                        sign = 1 if board.turn else -1
                        gap = sign * (good_label['score_cp'] - bad_label['score_cp'])
                        if gap >= 50 and abs(good_label['score_cp']) < 1500 and abs(bad_label['score_cp']) < 1500:
                            pairs.append({'good_fen': good.fen(), 'bad_fen': bad.fen(), 'sign': sign,
                                          'game_id': root['game_id'], 'cp_loss': gap, 'source': 'budgeted_teacher_candidates'})
                if roots % 25 == 0:
                    print(f'{split}: {len(added)}/{target} labels, {len(pairs)} pairs, {refined} refinements', flush=True)
            (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows + added))
            (pairdir / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pairs))
            counts[split] = {'base': len(rows), 'new': len(added), 'pairs': len(pairs),
                             'refined': refined, 'roots': roots, 'sources': sources}
            print(split, counts[split], flush=True)
    manifest = {'seed': 719, 'counts': counts, 'elapsed_seconds': time.monotonic() - started,
                'requested_stockfish_nodes': total_budget,
                'label_policy': '500 then 1500 nodes; additional 6000 for unstable, close or student-disagreement examples, at most 25% of new quota. Node budgets are cooperative.',
                'split_policy': 'Inherited whole-game splits; canonical FEN/color-mirror owners exclude cross-split aliases, including ranking pairs.',
                'source_manifest_sha256': hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
                'stockfish_sha256': hashlib.sha256(args.stockfish.read_bytes()).hexdigest(),
                'student_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (pairdir / 'manifest.json').write_text(json.dumps({'counts': {s: v['pairs'] for s,v in counts.items()},
        'parent_manifest_sha256': hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
