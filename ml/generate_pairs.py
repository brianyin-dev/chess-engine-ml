"""Generate mover-relative better/worse candidate pairs with Stockfish labels."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import chess
import chess.engine
from benchmarks.analyze import review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--stockfish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    rng = random.Random(331)
    base = {s: [json.loads(l) for l in (args.data / f'{s}.jsonl').read_text().splitlines()]
            for s in ('train', 'val', 'test')}
    fens = {s: {r['fen'] for r in records} for s, records in base.items()}
    pairs = {s: [] for s in base}
    seen = set()
    def add(split, pair):
        key = (pair['good_fen'], pair['bad_fen'])
        if key in seen or any(pair[f'{kind}_fen'] in fens[other]
                             for kind in ('good', 'bad') for other in base if other != split):
            return
        # Also prevent any identical successor position crossing pair splits.
        if any(pair[f'{kind}_fen'] in {p[f'{k}_fen'] for p in pairs[other] for k in ('good','bad')}
               for kind in ('good','bad') for other in pairs if other != split):
            return
        pairs[split].append(pair)
        seen.add(key)
    for pair in json.loads(args.analysis.read_text())['pairs']:
        add('train', pair)
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve())) as engine:
        engine.configure({'Threads': 1, 'Hash': 32})
        for split, records in base.items():
            for parent in rng.sample(records, min(len(records), 600 if split == 'train' else 150)):
                board = chess.Board(parent['fen'])
                legal = list(board.legal_moves)
                if not legal:
                    continue
                move = rng.choice(legal)
                row = review(engine, board, move, .08)
                if not (row['allows_mate'] or row['missed_forced_mate'] or (row['cp_loss'] or 0) >= 50):
                    continue
                good, bad = board.copy(), board.copy()
                good.push_uci(row['best_move'])
                bad.push(move)
                add(split, {'good_fen': good.fen(), 'bad_fen': bad.fen(),
                            'sign': 1 if board.turn else -1, 'game_id': parent['game_id'],
                            'cp_loss': row['cp_loss'], 'source': 'sampled_candidate_comparison'})
            print(split, len(pairs[split]), 'pairs', flush=True)
    args.output.mkdir(parents=True)
    for split, records in pairs.items():
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (args.output / 'manifest.json').write_text(json.dumps({
        'seed': 331, 'analysis_ms_per_root': 80, 'minimum_cp_loss': 50,
        'analysis_sha256': hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        'data_manifest_sha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
        'stockfish_sha256': hashlib.sha256(args.stockfish.read_bytes()).hexdigest(),
        'counts': {s: len(r) for s, r in pairs.items()},
        'policy': 'Better/worse successor boards, signed for the root mover. '
                  'First mistakes from old lost games are training only. '
                  'No identical successor FEN crosses base or pair splits.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
