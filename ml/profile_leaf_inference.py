"""Measure identical NN scores with and without redundant search-leaf checks."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import chess
from engine.search import search
from ml.evaluator import NeuralEvaluator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('output must be new')
    fast = NeuralEvaluator(args.checkpoint, .25, quiet_only=True)
    reference = NeuralEvaluator(args.checkpoint, .25, quiet_only=True, optimized=False)
    boards = [chess.Board(json.loads(line)['fen']) for line in args.data.read_text().splitlines()[:1024]]
    boards = [b for b in boards if not b.is_game_over()]
    assert all(fast.evaluate_position(b) == reference.evaluate_position(b) for b in boards)
    trials = {'reference': [], 'fast': []}
    for repeat in range(5):
        for name, evaluator in ([('reference', reference), ('fast', fast)] if repeat % 2 else
                                [('fast', fast), ('reference', reference)]):
            start = perf_counter()
            for board in boards: evaluator.evaluate_position(board)
            trials[name].append((perf_counter() - start) * 1000 / len(boards))
    search_rows = []
    for fen in [chess.STARTING_FEN,
                'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7']:
        before = search(chess.Board(fen), depth=3, eval_fn=reference)
        after = search(chess.Board(fen), depth=3, eval_fn=fast)
        assert (before.score, before.move, before.nodes, before.qnodes) == (after.score, after.move, after.nodes, after.qnodes)
        search_rows.append({'fen': fen, 'reference_seconds': before.elapsed, 'fast_seconds': after.elapsed,
                            'score': after.score, 'move': after.move.uci(), 'visited_nodes': after.nodes + after.qnodes})
    report = {'boards': len(boards), 'rounded_score_mismatches': 0,
              'policy': 'Nonterminal search-leaf scores; alternating five trials; same v8 checkpoint, .25 quiet correction.',
              'trials_ms_per_evaluation': trials,
              'median_ms_per_evaluation': {k: median(v) for k, v in trials.items()},
              'fixed_depth_search': search_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
