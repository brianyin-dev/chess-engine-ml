"""Predeclared common-opening correction sweep; fresh confirmation if promising."""
import json
from pathlib import Path
import subprocess
import sys
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'ml/artifacts/weight-control-v16'
CP = 'ml/artifacts/quiet-ranking-v8.pt'


def main():
    ART.mkdir(parents=True, exist_ok=True)
    state = ART / 'status.json'
    if state.exists():
        raise FileExistsError('Preserve existing results')

    def status(phase, **extra):
        temp = state.with_suffix('.tmp')
        temp.write_text(json.dumps({'phase': phase, **extra}, indent=2) + '\n')
        temp.replace(state)
        print(phase, flush=True)

    def run(phase, args):
        status(phase, command=args)
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True)

    common = ['-m', 'ml.compare', '--checkpoint', CP, '--quiet-only',
              '--opponent-checkpoint', CP, '--opponent-nn-weight', '0',
              '--opponent-quiet-only', '--opponent-incremental',
              '--time-ms', '250', '--max-plies', '1000']
    protocol = {
        'reference': 'Incremental handcrafted evaluator, 0% NN correction',
        'checkpoint': CP, 'seed': 162500, 'time_ms': 250,
        'pairs_per_screen': 10, 'depth_cap': 8,
        'screens': ['original-heuristic', 'old-NN-25', 'weight-05', 'weight-10', 'weight-25'],
        'order': 'Predeclared sequential screens; identical opening order, paired colors',
        'confirmation': 'Best positive weight (ties prefer smaller) requires 60% and 20 complete error-free pilot games. Confirm on 20 fresh games; if >=60%, extend independent confirmation to 100 games. Selection pilot excluded from confirmation.',
        'promotion': 'No automatic app promotion or Stockfish matches',
    }
    (ART / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    try:
        with exclusive_cpu('controlled correction-weight sweep'):
            run('freeze-openings', ['-m', 'benchmarks.generate_strength_openings',
                '--output', 'benchmarks/openings-weight-v16.json', '--pairs', '60',
                '--seed', '162500', '--exclude-data', 'ml/data/aligned-v11-2026'])
            starts = json.loads((ROOT / 'benchmarks/openings-weight-v16.json').read_text())
            for name, items in [('pilot', starts[:10]), ('confirmation', starts[10:20]), ('extension', starts[20:])]:
                (ROOT / f'benchmarks/openings-weight-v16-{name}.json').write_text(json.dumps(items, indent=2) + '\n')
            results = {}
            for name, weight, incremental in [('original-heuristic', '0', False), ('old-NN-25', '.25', False),
                                              ('weight-05', '.05', True), ('weight-10', '.10', True), ('weight-25', '.25', True)]:
                output = ART / name
                run(name, [*common, '--nn-weight', weight, *(['--incremental'] if incremental else []),
                    '--openings', 'benchmarks/openings-weight-v16-pilot.json', '--pairs', '10', '--output', str(output)])
                results[name] = json.loads((output / 'report.json').read_text())['summary']
                (ART / 'screen-summary.json').write_text(json.dumps(results, indent=2) + '\n')
            weights = [('weight-05', '.05'), ('weight-10', '.10'), ('weight-25', '.25')]
            best, weight = max(weights, key=lambda item: results[item[0]]['score_fraction_completed'])
            def encouraging(summary, count):
                return summary['completed'] == count and not summary['errors'] and summary['score_fraction_completed'] >= .6
            confirmed = False
            if encouraging(results[best], 20):
                output = ART / 'confirmation'
                run('confirmation', [*common, '--nn-weight', weight, '--incremental', '--openings',
                    'benchmarks/openings-weight-v16-confirmation.json', '--pairs', '10', '--output', str(output)])
                report = json.loads((output / 'report.json').read_text())
                results['confirmation'] = report['summary']
                if encouraging(report['summary'], 20):
                    output = ART / 'confirmation-extension'
                    run('confirmation-extension', [*common, '--nn-weight', weight, '--incremental', '--openings',
                        'benchmarks/openings-weight-v16-extension.json', '--pairs', '40', '--output', str(output)])
                    extension = json.loads((output / 'report.json').read_text())['games']
                    for game in extension:
                        game['pair'] += 10
                    results['confirmation-100'] = summarize(report['games'] + extension)
                    confirmed = True
            status('completed', results=results, selected_weight=weight, confirmation_100=confirmed, protocol=protocol)
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
