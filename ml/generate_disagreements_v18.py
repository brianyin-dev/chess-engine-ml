"""Collect fresh game-separated consequential disagreements at actual search leaves."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import chess
import chess.engine
from benchmarks.analyze import review
from engine.evaluation import evaluate
from engine.search import search
from ml.audit import SampledEvaluator
from ml.evaluator import NeuralEvaluator
from ml.generate_search_data import key

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'ml/artifacts/disagreements-v18'
DATA = ROOT / 'ml/data/disagreements-v18-2026'
OLD = ROOT / 'ml/artifacts/quiet-ranking-v8.pt'
BASE = ROOT / 'ml/data/aligned-v17-2026'
SF = ROOT / 'tools/stockfish-sf19/stockfish/stockfish-macos-universal'


def source_split(index, student=False):
    split = 'train' if index % 6 < 4 else 'val' if index % 6 == 4 else 'test'
    if student and index // 6 % 2 and split != 'train':
        split = 'test' if split == 'val' else 'val'
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--openings-file', type=Path, default=ROOT/'benchmarks/openings-disagreements-v18.json')
    parser.add_argument('--game-offset', type=int, default=0)
    parser.add_argument('--confirmation-threshold-cp', type=int, default=100)
    parser.add_argument('--probe-nodes', type=int, default=600)
    parser.add_argument('--play-policy', choices=['teacher','student'], default='teacher')
    parser.add_argument('--game-plies', type=int, default=48)
    args = parser.parse_args()
    if DATA.exists() and not args.append: raise FileExistsError('Preserve dataset')
    previous = json.loads((DATA/'manifest.json').read_text()) if args.append else None
    if min(args.probe_nodes, args.confirmation_threshold_cp, args.game_plies) < 1 or args.game_offset < 0:
        parser.error('positive budgets and nonnegative game offset required')
    if previous and args.game_offset != sum(batch['games'] for batch in previous.get('batches', [{'games':120}])):
        parser.error('append offset must preserve all preceding source game IDs')
    DATA.mkdir(parents=True, exist_ok=True); (DATA / 'pairs').mkdir(exist_ok=True)
    rng = random.Random(180001 + args.game_offset)
    seen = set()
    for split in ('train', 'val', 'test'):
        for path in (BASE / f'{split}.jsonl', BASE / 'pairs' / f'{split}.jsonl'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                seen.update(key(row[f]) for f in ('fen', 'good_fen', 'bad_fen', 'root_fen') if f in row)
    starts = json.loads(args.openings_file.read_text())
    model = NeuralEvaluator(OLD, .25, True, incremental=True)
    zero = NeuralEvaluator(OLD, 0, True, incremental=True)
    rows = {s: {} for s in ('train', 'val', 'test')}
    pairs = {s: [] for s in rows}
    owners = {}
    pair_seen = set()
    stats = {'quiet_search_leaves': 0, 'disagreement_positions': 0, 'confirmed_100cp_disagreements': 0,
             'NN_harmed_heuristic': 0, 'unattainable_train_pairs': 0, 'teacher_root_reviews': 0}
    stats['confirmed_disagreements'] = 0
    stats['confirmed_50_to_99cp_disagreements'] = 0
    if args.append:
        for split in rows:
            records = [json.loads(line) for line in (DATA/f'{split}.jsonl').read_text().splitlines()]
            rows[split] = {key(r['fen']):r for r in records}
            pairs[split] = [json.loads(line) for line in (DATA/'pairs'/f'{split}.jsonl').read_text().splitlines()]
            for row in records + pairs[split]:
                seen.update(key(row[f]) for f in ('fen','root_fen','good_fen','bad_fen') if f in row)
            pair_seen.update((key(r['good_fen']),key(r['bad_fen'])) for r in pairs[split])
        for name,value in previous['statistics'].items():
            stats[name] = value
        stats['confirmed_disagreements'] = previous['statistics'].get('confirmed_disagreements', previous['statistics']['confirmed_100cp_disagreements'])
    started = time.monotonic()
    with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads': 1, 'Hash': 32})
        def settle(board):
            board = board.copy(stack=False); line = []
            for _ in range(17):
                if board.is_game_over(): return None
                info = sf.analyse(board, chess.engine.Limit(nodes=12000), game=object())
                move = info['pv'][0]
                if not (board.is_check() or board.is_capture(move) or move.promotion):
                    cp = info['score'].white().score(mate_score=10000)
                    if abs(cp) >= 1500: return None
                    return board, cp, line
                line.append(move.uci()); board.push(move)
            return None
        for index, opening in enumerate(starts):
            absolute_index = index + args.game_offset
            split = source_split(absolute_index, args.play_policy == 'student')
            game_id = 6000000 + absolute_index
            board = chess.Board()
            for move in opening['moves']: board.push_uci(move)
            for ply in range(args.game_plies):
                if board.is_game_over(): break
                if ply % 8 == 0:
                    sample = SampledEvaluator(model, rng, 24)
                    search(board, depth=3, node_limit=1000, eval_fn=sample)
                    quiet = [chess.Board(fen) for fen in dict.fromkeys(sample.fens)]
                    quiet = [b for b in quiet if not b.is_game_over() and not b.is_check()
                             and not next(b.generate_legal_captures(), None) and key(b.fen()) not in seen]
                    rng.shuffle(quiet)
                    for leaf in quiet[:3]:
                        root_key = key(leaf.fen())
                        if root_key in seen or owners.get(root_key, split) != split: continue
                        stats['quiet_search_leaves'] += 1
                        h = search(leaf, depth=3, node_limit=args.probe_nodes, eval_fn=zero)
                        n = search(leaf, depth=3, node_limit=args.probe_nodes, eval_fn=model)
                        if h.move == n.move: continue
                        stats['disagreement_positions'] += 1
                        # Independently restricted-root teacher reviews; fresh engine state.
                        nr = review(sf, leaf, n.move, .1)
                        hr = review(sf, leaf, h.move, .1)
                        stats['teacher_root_reviews'] += 2
                        if not any((r['cp_loss'] or 0) >= max(25, args.confirmation_threshold_cp-25) or r['allows_mate'] or r['missed_forced_mate'] for r in (nr, hr)): continue
                        nr = review(sf, leaf, n.move, .4); hr = review(sf, leaf, h.move, .4)
                        stats['teacher_root_reviews'] += 2
                        if not any((r['cp_loss'] or 0) >= args.confirmation_threshold_cp or r['allows_mate'] or r['missed_forced_mate'] for r in (nr, hr)): continue
                        stats['confirmed_disagreements'] += 1
                        large = any((r['cp_loss'] or 0) >= 100 or r['allows_mate'] or r['missed_forced_mate'] for r in (nr,hr))
                        stats['confirmed_100cp_disagreements'] += large
                        stats['confirmed_50_to_99cp_disagreements'] += not large
                        harmed = (nr['cp_loss'] or 0) >= args.confirmation_threshold_cp and (hr['cp_loss'] or 0) < args.confirmation_threshold_cp/2 and not hr['allows_mate']
                        stats['NN_harmed_heuristic'] += harmed
                        best = chess.Move.from_uci(nr['best_move'])
                        a = leaf.copy(); a.push(best); good = settle(a)
                        if not good: continue
                        for move, reviewed in ((n.move, nr), (h.move, hr)):
                            if move == best: continue
                            b = leaf.copy(); b.push(move); bad = settle(b)
                            if not bad: continue
                            sign = 1 if leaf.turn else -1
                            gap = sign * (good[1] - bad[1])
                            endpoint_keys = (key(good[0].fen()), key(bad[0].fen()))
                            if gap < 50 or endpoint_keys[0] == endpoint_keys[1] or any(k in seen or owners.get(k, split) != split for k in endpoint_keys): continue
                            factors = [0 if next(x.generate_legal_captures(), None) or x.is_check() else .25 for x in (good[0], bad[0])]
                            if not sum(factors): continue
                            base_gap = sign * (evaluate(good[0]) - evaluate(bad[0]))
                            attainable = base_gap + 250 * sum(factors) > 5
                            if split == 'train' and not attainable:
                                stats['unattainable_train_pairs'] += 1; continue
                            if endpoint_keys in pair_seen: continue
                            pair_seen.add(endpoint_keys)
                            record = {'root_fen': leaf.fen(), 'good_fen': good[0].fen(), 'bad_fen': bad[0].fen(),
                                      'sign': sign, 'game_id': game_id, 'cp_loss': gap, 'heuristic_gap_cp': base_gap,
                                      'good_move': best.uci(), 'bad_move': move.uci(), 'good_tactical_line': good[2],
                                      'bad_tactical_line': bad[2], 'source': 'search_disagreement',
                                      'NN_harmed_heuristic': harmed, 'attainable': attainable,
                                      'confirmation_threshold_cp': args.confirmation_threshold_cp, 'probe_nodes': args.probe_nodes,
                                      'training_weight': 4 if split == 'train' and harmed else 2 if split == 'train' else 1,
                                      'heuristic_review': hr, 'NN_review': nr}
                            pairs[split].append(record)
                            owners[root_key] = split
                            for k, item in zip(endpoint_keys, (good, bad)):
                                owners[k] = split
                                rows[split][k] = {'fen': item[0].fen(), 'score_cp': item[1], 'game_id': game_id,
                                                  'ply': item[0].ply(), 'root_fen': leaf.fen(), 'source': 'search_disagreement_settled'}
                if args.play_policy == 'student':
                    result = search(board, depth=8, node_limit=3000,
                                    eval_fn=model if (board.turn == chess.WHITE) == (index % 2 == 0) else zero)
                    move = result.move
                else:
                    move = sf.play(board, chess.engine.Limit(nodes=8000), game=index).move
                if move is None: break
                board.push(move)
            # Assign roots/endpoints permanently to one source game as well as split.
            seen.update(owners); owners.clear()
            for s in rows:
                (DATA / f'{s}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[s].values()))
                (DATA / 'pairs' / f'{s}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs[s]))
            print(f'Game {index+1}/{len(starts)}: '+str({s:len(v) for s,v in pairs.items()}), flush=True)
    manifest = {'policy': 'Fresh weighted book starts, teacher or unchanged student self-play as recorded per batch. Game IDs split4train/1val/1test; quiet actual search leaves sampled, compare equal node-limited depth3 searches with 0% and25% evaluators; budgets and confirmation thresholds recorded per batch. Confirm differences with full-strength Stockfish400ms root analysis; initial threshold100cp, supplemental50cp. settle endpoints12knodes. Exclude all previous score/pair/root canonical aliases. Keep both NN-harm and NN-help examples; no model-based validation/test filtering. Train excludes corrections incapable of5cp margin; holdouts retain them. All endpoints require at leastone active quiet gate. Val/test teacher-disagreement distribution is targeted, not general chess strength.',
                'counts': {s:{'rows':len(rows[s]),'pairs':len(pairs[s])} for s in rows}, 'statistics': stats,
                'elapsed_seconds': time.monotonic()-started, 'seed':180001 + args.game_offset,
                'batches': previous.get('batches',[{'games':120,'confirmation_cp':100,'probe_nodes':600,'seed':180001}]) + [{'games':len(starts),'confirmation_cp':args.confirmation_threshold_cp,'probe_nodes':args.probe_nodes,'seed':180001 + args.game_offset,'play_policy':args.play_policy,'game_plies':args.game_plies}] if previous else [{'games':len(starts),'confirmation_cp':args.confirmation_threshold_cp,'probe_nodes':args.probe_nodes,'seed':180001}],
                'checkpoint_sha256':hashlib.sha256(OLD.read_bytes()).hexdigest(),
                'openings_sha256':hashlib.sha256(args.openings_file.read_bytes()).hexdigest()}
    for path in (DATA/'manifest.json',DATA/'pairs'/'manifest.json'):
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    if not all(pairs.values()): raise ValueError('Empty disagreement split')


if __name__ == '__main__': main()
