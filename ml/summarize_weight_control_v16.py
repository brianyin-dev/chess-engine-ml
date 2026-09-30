"""Audit common conditions and summarize opening-pair uncertainty."""
import json
import random
from pathlib import Path

ART = Path(__file__).resolve().parent / 'artifacts/weight-control-v16'
NAMES = ['original-heuristic', 'old-NN-25', 'weight-05', 'weight-10', 'weight-25']


def pair_scores(report):
    pairs = {}
    for game in report['games']:
        score = {'win': 1., 'draw': .5, 'loss': 0.}.get(game['current_result'])
        if score is None:
            raise ValueError('Incomplete games cannot enter paired comparison')
        pairs.setdefault(game['pair'], []).append(score)
    if any(len(scores) != 2 for scores in pairs.values()):
        raise ValueError('Each opening needs two completed games')
    return {pair: sum(scores) / 2 for pair, scores in pairs.items()}


def interval(values):
    # Resample entire opening pairs, retaining correlation between colors.
    rng = random.Random(162501)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(20000))
    return [samples[500], samples[19499]]


def main():
    reports = {name: json.loads((ART / name / 'report.json').read_text()) for name in NAMES}
    reference = reports[NAMES[0]]
    for report in reports.values():
        for field in ['checkpoint_sha256', 'openings_sha256', 'engine_sources_sha256', 'evaluation_sources_sha256', 'opponent']:
            if report[field] != reference[field]:
                raise ValueError(f'Different common condition: {field}')
        for field in ['time_ms', 'max_plies', 'depth_cap', 'quiet_only', 'node_limit']:
            if report['config'][field] != reference['config'][field]:
                raise ValueError(f'Different common setting: {field}')
        if not report.get('finished_at'):
            raise ValueError('Screen still running')
        actual = [(g['pair'], g['initial_fen'], g['current_color']) for g in report['games']]
        expected = [(g['pair'], g['initial_fen'], g['current_color']) for g in reference['games']]
        if actual != expected:
            raise ValueError('Different opening/color pairings')
    pairs = {name: pair_scores(report) for name, report in reports.items()}
    summary = {'policy': 'Descriptive percentile bootstrap, 20,000 resamples of 10 opening pairs, seed 162501. Small screening sample; intervals do not correct for selecting among weights. Common-opponent comparisons are not direct head-to-head matches.', 'screens': {}, 'paired_score_differences': {}}
    for name, report in reports.items():
        summary['screens'][name] = {**report['summary'], 'pair_bootstrap_95_interval': interval(list(pairs[name].values()))}
    for name in NAMES[2:]:
        for control in NAMES[:2]:
            differences = [pairs[name][pair] - pairs[control][pair] for pair in pairs[name]]
            summary['paired_score_differences'][f'{name} minus {control}'] = {'mean': sum(differences) / len(differences), 'pair_bootstrap_95_interval': interval(differences)}
    (ART / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
