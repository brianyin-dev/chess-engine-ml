"""Confirmed mistakes, runtime-aligned training, independent validation and games."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'ml/artifacts/targeted-v17'
OLD = 'ml/artifacts/quiet-ranking-v8.pt'
SF = 'tools/stockfish-sf19/stockfish/stockfish-macos-universal'
BASE = 'ml/data/aligned-v11-2026'
NEW = 'ml/data/confirmed-v17-2026'
DATA = 'ml/data/aligned-v17-2026'
CP = 'ml/artifacts/targeted-v17/trained.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-small-holdout', action='store_true')
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    if (ART / 'status.json').exists() and not args.resume_small_holdout: raise FileExistsError('Preserve experiment')
    if args.resume_small_holdout:
        prior = json.loads((ART / 'status.json').read_text())
        small = json.loads((ART / 'fresh-holdout.json').read_text())['rows']
        if prior['phase'] != 'completed' or max(map(len, small.values())) >= 30 or (ART / 'fresh-validation-small.json').exists():
            raise ValueError('Resume only the completed undersized-holdout run once')
        (ART / 'status.json').rename(ART / 'status-small.json')
        (ART / 'fresh-validation.json').rename(ART / 'fresh-validation-small.json')
    def status(phase, **extra):
        tmp = ART / 'status.tmp'; tmp.write_text(json.dumps({'phase': phase, **extra}, indent=2) + '\n')
        tmp.replace(ART / 'status.json'); print(phase, flush=True)
    def run(phase, args):
        status(phase, command=args); subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    protocol = {'training': 'Unchanged v8 architecture, .25 quiet-only residual blend, rounded full endpoint scores with straight-through gradients. Retain prior labels/rankings, prioritize confirmed errors, protect heuristic-correct rankings and anchor original corrections. One candidate only.',
                'validation': 'If initial quiet-only holdout has fewer than30 pairs per split, expand teacher sampling with fixed random alternatives and at leastone quiet endpoint, excluding all training aliases, before candidate scoring. No retraining. Fresh teacher self-play from 30 book starts; opening-separated val/test; prior data/reviewed game aliases excluded. Freeze before training and reserve from new labels. Strict validation ranking gain, no extra regressions; test nonregression. No tuning on fresh test.',
                'games': 'Only after fresh gate passes: 20 paired games against optimized 0%, then 20 against unchanged old nonincremental .25 quiet NN, same fresh starts, 250ms, depth8, 1000maxplies. If >=60% vs0 and >50% vsold with all complete/noerrors, extend vs0 to100 total games on40 fresh pairs. Report pilot and extension separately. No Stockfish or app promotion.'}
    (ART / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    try:
        with exclusive_cpu('v17 targeted training and isolated games'):
            if not args.resume_small_holdout:
                failures = json.loads((ART / 'failures.json').read_text())
                if 'summary' not in failures: raise ValueError('Wait for completed review')
                run('tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-q'])
                run('freeze-independent-positions', ['-m', 'benchmarks.generate_strength_openings', '--output',
                    'benchmarks/openings-targeted-v17-holdout.json', '--pairs', '30', '--seed', '170001', '--exclude-data', BASE])
                run('label-independent-positions', ['-m', 'ml.fresh_holdout_v17'])
                run('generate-confirmed-training', ['-m', 'ml.generate_quiet_rankings', '--data', BASE,
                    '--checkpoint', OLD, '--failures', str(ART / 'failures.json'), '--stockfish', SF,
                    '--output', NEW, '--train-pairs', '800', '--heldout-pairs', '120', '--failure-id-offset', '4000000',
                    '--reserved-positions', str(ART / 'fresh-holdout.json')])
                run('retain-existing-data', ['-m', 'ml.merge_rankings', '--old', BASE, '--new', NEW, '--output', DATA])
                run('runtime-aligned-training', ['-m', 'ml.train', '--data', DATA, '--pairs', DATA + '/pairs',
                    '--checkpoint', CP, '--metrics', str(ART / 'training.json'), '--features', 'relationships',
                    '--target', 'residual', '--color-consistent', '--correction-limit-cp', '250', '--correction-weight', '.25',
                    '--quiet-only', '--initial-checkpoint', OLD, '--learning-rate', '.0001', '--rank-weight', '4',
                    '--rank-margin-cp', '5', '--hard-pair-weight', '2', '--protected-pair-weight', '3',
                    '--score-weight', '.1', '--anchor-weight', '.5', '--selection', 'ranking', '--epochs', '25', '--patience', '6'])
            run('independent-validation', ['-m', 'ml.validate_targeted_v17'])
            validation = json.loads((ART / 'fresh-validation.json').read_text())
            if not validation['eligible_for_games']:
                status('completed', fresh_validation=validation, games_skipped='Independent improvement gate failed',
                       protocol=protocol, app_promoted=False)
                return
            run('freeze-independent-games', ['-m', 'benchmarks.generate_strength_openings', '--output',
                'benchmarks/openings-targeted-v17-games.json', '--pairs', '50', '--seed', '172500', '--exclude-data', DATA])
            starts = json.loads((ROOT / 'benchmarks/openings-targeted-v17-games.json').read_text())
            for name, items in [('pilot', starts[:10]), ('extension', starts[10:])]:
                (ROOT / f'benchmarks/openings-targeted-v17-games-{name}.json').write_text(json.dumps(items, indent=2) + '\n')
            common = ['-m', 'ml.compare', '--checkpoint', CP, '--nn-weight', '.25', '--quiet-only', '--incremental',
                '--opponent-checkpoint', OLD, '--opponent-quiet-only', '--openings',
                'benchmarks/openings-targeted-v17-games-pilot.json', '--pairs', '10', '--time-ms', '250', '--max-plies', '1000']
            results = {}
            for name, extra in [('vs-optimized-zero', ['--opponent-nn-weight', '0', '--opponent-incremental']),
                                 ('vs-old-NN', ['--opponent-nn-weight', '.25'])]:
                output = ART / name; run(name, [*common, *extra, '--output', str(output)])
                results[name] = json.loads((output / 'report.json').read_text())['summary']
            eligible = (all(s['completed'] == 20 and not s['errors'] for s in results.values())
                        and results['vs-optimized-zero']['score_fraction_completed'] >= .6
                        and results['vs-old-NN']['score_fraction_completed'] > .5)
            if eligible:
                output = ART / 'vs-optimized-zero-extension'
                run('extend-zero-to-100', ['-m', 'ml.compare', '--checkpoint', CP, '--nn-weight', '.25', '--quiet-only',
                    '--incremental', '--opponent-checkpoint', OLD, '--opponent-nn-weight', '0', '--opponent-quiet-only',
                    '--opponent-incremental', '--openings', 'benchmarks/openings-targeted-v17-games-extension.json',
                    '--pairs', '40', '--time-ms', '250', '--max-plies', '1000', '--output', str(output)])
                initial = json.loads((ART / 'vs-optimized-zero/report.json').read_text())['games']
                extension = json.loads((output / 'report.json').read_text())['games']
                for game in extension: game['pair'] += 10
                results['vs-optimized-zero-100'] = summarize(initial + extension)
            status('completed', fresh_validation=validation, matches=results, extended_to_100=eligible,
                   protocol=protocol, app_promoted=False)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__': main()
