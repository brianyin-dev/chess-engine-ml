"""Expand split-isolated self-play data with independently labeled search calls."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import chess
import chess.engine
from engine.search import search
from ml.audit import SampledEvaluator, reference
from ml.evaluator import NeuralEvaluator


def key(fen):
    board = chess.Board(fen)
    return min(' '.join(board.fen().split()[:4]), ' '.join(board.mirror().fen().split()[:4]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stockfish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roots-per-split', type=int, default=40)
    parser.add_argument('--samples-per-root', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or min(args.roots_per_split, args.samples_per_root) < 1:
        parser.error('use a new output directory and positive sample counts')
    rng = random.Random(461)
    splits = ('train', 'val', 'test')
    rows = {s: [json.loads(l) for l in (args.data / f'{s}.jsonl').read_text().splitlines()] for s in splits}
    # Assign canonical mirrored positions to one split; remove preexisting aliases.
    owners = {}
    for s in ('test', 'val', 'train'):
        kept = []
        for row in rows[s]:
            k = key(row['fen'])
            if k not in owners:
                owners[k] = s
                kept.append(row)
        rows[s] = kept
    evaluator = NeuralEvaluator(args.checkpoint)
    args.output.mkdir(parents=True)
    counts = {}
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve())) as oracle:
        oracle.configure({'Threads': 1, 'Hash': 64})
        for split in splits:
            candidates = [r for r in rows[split] if not chess.Board(r['fen']).is_game_over()]
            roots = rng.sample(candidates, min(len(candidates), args.roots_per_split))
            added = []
            for i, root in enumerate(roots, 1):
                sampler = SampledEvaluator(evaluator, rng, args.samples_per_root)
                search(chess.Board(root['fen']), depth=3, eval_fn=sampler)
                for fen in sampler.fens:
                    k = key(fen)
                    if k in owners:
                        continue
                    owners[k] = split
                    label = reference(oracle, chess.Board(fen))
                    added.append({'fen': fen, 'score_cp': label['score_cp'],
                                  'game_id': root['game_id'], 'ply': root['ply'],
                                  'source': 'search_stand_pat', 'root_fen': root['fen']})
                print(f'{split} root {i}/{len(roots)}: {len(added)} new labels', flush=True)
            counts[split] = {'base': len(rows[split]), 'search': len(added)}
            (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[split] + added))
    manifest = {'seed': 461, 'counts': counts, 'depth': 3, 'label_depth': 10,
                'policy': 'Whole-game inherited splits; canonical FEN and color-mirror aliases excluded across splits. Search calls are not all quiet leaves.',
                'source_manifest_sha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
    provenance = {'roots_per_split': args.roots_per_split,
                  'samples_per_root': args.samples_per_root,
                  'stockfish_sha256': hashlib.sha256(args.stockfish.read_bytes()).hexdigest()}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
