"""One-shot fixed-blend validation on a frozen independent ranking holdout."""
import json
import subprocess
import sys
from pathlib import Path
import hashlib
from ml.evaluator import NeuralEvaluator
from ml.evaluate_blends import pair_metrics

ART = Path(__file__).resolve().parent / 'artifacts/targeted-v17'
OLD = Path('ml/artifacts/quiet-ranking-v8.pt')


def main():
    output = ART / 'fresh-validation.json'
    if output.exists(): raise FileExistsError('Preserve one-shot test')
    holdout = ART / 'fresh-holdout.json'
    cases = json.loads(holdout.read_text())['rows']
    if min(map(len, cases.values())) < 30:
        # Freeze more independently labeled cases before loading/scoring candidate.
        subprocess.run([sys.executable, '-m', 'benchmarks.generate_strength_openings', '--output',
                        'benchmarks/openings-targeted-v17-holdout-expanded.json', '--pairs', '60',
                        '--seed', '170002', '--exclude-data', 'ml/data/aligned-v17-2026'], check=True)
        subprocess.run([sys.executable, '-m', 'ml.fresh_holdout_v17', '--expanded'], check=True)
        holdout = ART / 'fresh-holdout-expanded.json'
        cases = json.loads(holdout.read_text())['rows']
    if min(map(len, cases.values())) < 30:
        raise ValueError('Insufficient independent holdout size for game qualification')
    models = {'heuristic': NeuralEvaluator(OLD, 0, True, incremental=True),
              'old': NeuralEvaluator(OLD, .25, True, incremental=True),
              'candidate': NeuralEvaluator(ART / 'trained.pt', .25, True, incremental=True)}
    metrics = {name: {split: pair_metrics(rows, model) for split, rows in cases.items()}
               for name, model in models.items()}
    old, new = metrics['old'], metrics['candidate']
    eligible = (new['val']['accuracy'] > old['val']['accuracy']
                and new['val']['previously_correct_regressions'] <= old['val']['previously_correct_regressions']
                and new['test']['accuracy'] >= old['test']['accuracy']
                and new['test']['previously_correct_regressions'] <= old['test']['previously_correct_regressions'])
    report = {'policy': 'Fixed .25 quiet-only hybrid, incremental inference, rounded integer scores. Initial positions frozen before training; expansion frozen after training without changing candidate, excluding all data aliases. Initial tiny holdout result preserved separately; expanded holdout is a supplementary check. Strict fresh-validation accuracy improvement with no extra heuristic-correct regressions; test accuracy and regressions must not worsen. No retraining after reading this holdout.',
              'metrics': metrics, 'eligible_for_games': eligible,
              'candidate_sha256': hashlib.sha256((ART / 'trained.pt').read_bytes()).hexdigest(),
              'holdout_sha256': hashlib.sha256(holdout.read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
