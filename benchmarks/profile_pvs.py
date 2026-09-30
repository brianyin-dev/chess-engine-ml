"""Compare PVS with full-window search at the same completed depth."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import chess
from engine.search import _Search, search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    original = _Search.__init__
    rows = []
    for fixture in json.loads(Path('benchmarks/positions.json').read_text()):
        if fixture['id'] not in {'starting-position', 'middlegame', 'poisoned-pawn'}:
            continue
        trials = {False: [], True: []}
        for repeat in range(5):
            for enabled in ([False, True] if repeat % 2 else [True, False]):
                def initialize(worker, *a, **kw):
                    original(worker, *a, **kw)
                    worker.use_pvs = enabled
                with patch.object(_Search, '__init__', initialize):
                    start = perf_counter()
                    result = search(chess.Board(fixture['fen']), depth=args.depth)
                    trials[enabled].append({'seconds': perf_counter() - start,
                                            'visited_nodes': result.nodes + result.qnodes,
                                            'score': result.score, 'move': result.move.uci()})
        assert len({t['score'] for runs in trials.values() for t in runs}) == 1
        rows.append({'id': fixture['id'], 'full_window': trials[False], 'pvs': trials[True],
                     'full_window_median_seconds': median(t['seconds'] for t in trials[False]),
                     'pvs_median_seconds': median(t['seconds'] for t in trials[True])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'depth': args.depth, 'positions': rows}, indent=2) + '\n')
    for row in rows:
        print(row['id'], row['full_window_median_seconds'], row['pvs_median_seconds'])


if __name__ == '__main__':
    main()
