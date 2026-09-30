"""Freeze book-derived opening starts before measuring playing strength."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import chess
from engine.opening_book import choose_book_move
from benchmarks.match import opening_board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--book', type=Path, default=Path('books/gm2001.bin'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=91320)
    parser.add_argument('--exclude-data',type=Path,help='Exclude canonical aliases of labeled positions and ranking endpoints')
    args = parser.parse_args()
    if args.output.exists() or args.pairs < 1:
        parser.error('use a new output path and positive pair count')
    excluded = set()
    for path in Path('benchmarks').glob('openings*.json'):
        items = json.loads(path.read_text())
        if not isinstance(items, list):
            continue  # Manifests and other benchmark configurations are not starts.
        for item in items:
            board = opening_board(item)
            excluded.add(' '.join(board.fen().split()[:4]))
    rng = random.Random(args.seed)
    aliases = set()
    if args.exclude_data:
        from ml.generate_search_data import key
        for path in args.exclude_data.glob('**/*.jsonl'):
            for line in path.read_text().splitlines():
                item=json.loads(line)
                for field in ('fen','good_fen','bad_fen','root_fen'):
                    if field in item:aliases.add(key(item[field]))
    openings = []
    for attempt in range(10000):
        board, moves = chess.Board(), []
        length = rng.choice([10, 12, 14, 16])
        for _ in range(length):
            choice = choose_book_move(board, args.book, rng=rng)
            if choice is None:
                break
            moves.append(choice.move.uci())
            board.push(choice.move)
        key = ' '.join(board.fen().split()[:4])
        if len(moves) != length or key in excluded or board.is_game_over():
            continue
        if args.exclude_data:
            from ml.generate_search_data import key as canonical_key
            if canonical_key(board.fen()) in aliases:continue
        excluded.add(key)
        openings.append({'name': f'Frozen book start {len(openings) + 1:02d}', 'moves': moves})
        if len(openings) == args.pairs:
            break
    if len(openings) != args.pairs:
        raise RuntimeError('not enough unique book positions')
    args.output.write_text(json.dumps(openings, indent=2) + '\n')
    manifest = {'seed': args.seed, 'pairs': args.pairs, 'attempts': attempt + 1,
                'excluded_labeled_position_aliases': len(aliases),
                'book_sha256': hashlib.sha256(args.book.read_bytes()).hexdigest(),
                'openings_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'selection': 'Weighted book sampling; 10/12/14/16 plies; unique FENs; '
                             'exclude previous benchmark starts. No engine-score or game-result filtering. '
                             'Each start is played with both colors; no book after the prefix.'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
