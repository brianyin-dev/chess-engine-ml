"""Freeze fresh, game-separated teacher-settled ranking cases before training."""
import argparse
import random
import hashlib
import json
from pathlib import Path
import chess
import chess.engine
from engine.evaluation import evaluate
from ml.generate_search_data import key

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'ml/artifacts/targeted-v17'
SF = ROOT / 'tools/stockfish-sf19/stockfish/stockfish-macos-universal'
DATA = ROOT / 'ml/data/aligned-v11-2026'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expanded', action='store_true')
    args = parser.parse_args()
    output = ART / ('fresh-holdout-expanded.json' if args.expanded else 'fresh-holdout.json')
    rng = random.Random(170002)
    if output.exists():
        raise FileExistsError('Preserve independent holdout')
    excluded = set()
    for split in ('train', 'val', 'test'):
        for file in (DATA / f'{split}.jsonl', DATA / 'pairs' / f'{split}.jsonl'):
            for line in file.read_text().splitlines():
                row = json.loads(line)
                excluded.update(key(row[field]) for field in ('fen', 'good_fen', 'bad_fen', 'root_fen') if field in row)
    # Entire reviewed game positions are development data, never fresh holdouts.
    for name in ('old-NN-25', 'weight-25'):
        report = json.loads((ROOT / f'ml/artifacts/weight-control-v16/{name}/report.json').read_text())
        for game in report['games']:
            board = chess.Board(game['initial_fen'])
            for move in game['opening_moves'] + [m['uci'] for m in game['moves']]:
                excluded.add(key(board.fen()))
                board.push_uci(move)
            excluded.add(key(board.fen()))
    if args.expanded:
        for split in ('train', 'val', 'test'):
            for file in (ROOT / 'ml/data/aligned-v17-2026' / f'{split}.jsonl', ROOT / 'ml/data/aligned-v17-2026/pairs' / f'{split}.jsonl'):
                for line in file.read_text().splitlines():
                    row = json.loads(line)
                    excluded.update(key(row[field]) for field in ('fen', 'good_fen', 'bad_fen', 'root_fen') if field in row)
    openings_file = ROOT / ('benchmarks/openings-targeted-v17-holdout-expanded.json' if args.expanded else 'benchmarks/openings-targeted-v17-holdout.json')
    starts = json.loads(openings_file.read_text())
    rows = {'val': [], 'test': []}
    used = set(excluded)
    with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads': 1, 'Hash': 32})
        def settle(board):
            board = board.copy(stack=False)
            line = []
            for _ in range(17):
                if board.is_game_over(): return None
                info = sf.analyse(board, chess.engine.Limit(nodes=12000), game=object())
                move = info['pv'][0]
                if not (board.is_check() or board.is_capture(move) or move.promotion):
                    if not args.expanded and next(board.generate_legal_captures(), None): return None
                    cp = info['score'].white().score(mate_score=10000)
                    if abs(cp) >= 1500: return None
                    return board, cp, line
                line.append(move.uci()); board.push(move)
            return None
        for index, opening in enumerate(starts):
            split = 'val' if index % 2 == 0 else 'test'
            board = chess.Board(opening.get('fen', chess.STARTING_FEN))
            for move in opening.get('moves', []): board.push_uci(move)
            for ply in range(64):
                if board.is_game_over(): break
                if ply in (range(8, 64, 8) if args.expanded else (8, 24, 40, 56)) and key(board.fen()) not in used:
                    infos = sf.analyse(board, chess.engine.Limit(nodes=24000), multipv=3, game=object())
                    best = infos[0]['pv'][0]
                    good = board.copy(); good.push(best)
                    a = settle(good)
                    if a:
                        candidates = [info['pv'][0] for info in infos[1:]]
                        if args.expanded:
                            candidates += rng.sample(list(board.legal_moves), min(3, board.legal_moves.count()))
                        for bad_move in dict.fromkeys(candidates):
                            if bad_move == best: continue
                            bad = board.copy(); bad.push(bad_move)
                            b = settle(bad)
                            if not b: continue
                            if args.expanded and all(next(leaf.generate_legal_captures(), None) for leaf in (a[0], b[0])): continue
                            sign = 1 if board.turn else -1
                            gap = sign * (a[1] - b[1])
                            keys = {key(a[0].fen()), key(b[0].fen())}
                            if gap < 50 or len(keys) != 2 or keys & used: continue
                            rows[split].append({'root_fen': board.fen(), 'good_fen': a[0].fen(),
                                'bad_fen': b[0].fen(), 'sign': sign, 'cp_loss': gap,
                                'heuristic_gap_cp': sign * (evaluate(a[0]) - evaluate(b[0])),
                                'game_id': 5000000 + index, 'opening': opening['name'],
                                'good_move': best.uci(), 'bad_move': bad_move.uci(),
                                'good_tactical_line': a[2], 'bad_tactical_line': b[2]})
                            used.update(keys | {key(board.fen())})
                            break
                info = sf.play(board, chess.engine.Limit(nodes=12000), game=index)
                if info.move is None: break
                board.push(info.move)
            print(f'Fresh game {index+1}/{len(starts)}: val={len(rows["val"])} test={len(rows["test"])}', flush=True)
    if not all(rows.values()): raise ValueError('Empty fresh holdout')
    report = {'policy': ('Expanded cases require at least one quiet endpoint, add fixed-seed random alternatives, and exclude all new training data; supplementary check frozen after training. ' if args.expanded else '') + 'Fresh book starts followed by Stockfish self-play; entire openings assigned alternately to validation/test. Quiet teacher-settled endpoints; >=50cp gaps; canonical position/color-mirror aliases of prior data and reviewed games excluded. Frozen before candidate training; never enters optimizer or checkpoint selection.',
              'nodes_root': 24000, 'nodes_settling': 12000, 'rows': rows,
              'openings_sha256': hashlib.sha256(openings_file.read_bytes()).hexdigest(),
              'expanded': args.expanded,
              'expansion_policy': 'Original holdout too small. Fixed teacher and seeded diverse legal alternatives; require at least one quiet endpoint so .25 gate can act. Expanded before first candidate evaluation, after training; all training-data aliases excluded. No retraining after evaluation.' if args.expanded else None,
              'stockfish_sha256': hashlib.sha256(SF.read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
