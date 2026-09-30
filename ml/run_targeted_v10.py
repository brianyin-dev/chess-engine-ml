"""Sequential NN experiment; wait for the existing timed benchmark first."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/targeted-v10'
SF='tools/stockfish-sf19/stockfish/stockfish-macos-universal'
OLD='ml/data/quiet-rankings-v8-2026'
NEW='ml/data/mistakes-v10-2026'
DATA='ml/data/targeted-v10-2026'
CP='ml/artifacts/targeted-ranking-v10.pt'


def main():
    global ART, NEW, DATA, CP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='New identifier for a reproduction; use letters, digits, and hyphens')
    args = parser.parse_args()
    metrics = 'ml/artifacts/targeted-ranking-v10.json'
    if args.run_id:
        if not all(c.isascii() and (c.isalnum() or c == '-') for c in args.run_id):
            parser.error('run-id must contain only letters, digits, and hyphens')
        ART = ROOT / 'ml/artifacts' / args.run_id
        NEW = f'ml/data/{args.run_id}-mistakes'
        DATA = f'ml/data/{args.run_id}-merged'
        CP = f'ml/artifacts/{args.run_id}.pt'
        metrics = f'ml/artifacts/{args.run_id}.json'
    if (ART / 'status.json').exists():
        parser.error('experiment already exists; use a new --run-id to preserve results')
    ART.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):
        record={'phase':phase,**extra}
        temp=ART/'status.tmp'
        temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(ART/'status.json')
        print(phase,flush=True)
    def run(phase,args):
        status(phase,command=args)
        subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    status('waiting-for-stockfish-100-game-benchmark')
    report=ROOT/'ml/artifacts/strength-target-v9-100games-1320-250ms/report.json'
    while not json.loads(report.read_text()).get('finished_at'):
        time.sleep(2)
    try:
        run('tests',['-m','unittest','discover','-s','tests','-q'])
        run('inference-profile',['-m','ml.profile_leaf_inference','--checkpoint','ml/artifacts/quiet-ranking-v8.pt',
                                  '--data',OLD+'/test.jsonl','--output',str(ART/'inference-profile.json')])
        run('confirm-recent-blunders',['-m','ml.analyze_losses','--report',
            'ml/artifacts/quiet-ranking-v8-w25-vs-heuristic-250ms/report.json',
            '--checkpoint','ml/artifacts/quiet-ranking-v8.pt','--nn-weight','.25',
            '--stockfish',SF,'--output',str(ART/'loss-analysis.json')])
        run('settled-mistake-data',['-m','ml.generate_quiet_rankings','--data',OLD,
            '--checkpoint','ml/artifacts/quiet-ranking-v8.pt','--failures',str(ART/'loss-analysis.json'),
            '--stockfish',SF,'--output',NEW,'--train-pairs','1000','--heldout-pairs','200',
            '--failure-id-offset','2000000'])
        run('retain-old-supervision',['-m','ml.merge_rankings','--old',OLD,'--new',NEW,'--output',DATA])
        run('train',['-m','ml.train','--data',DATA,'--pairs',DATA+'/pairs',
            '--checkpoint',CP,'--metrics',metrics,
            '--features','relationships','--target','residual','--color-consistent',
            '--correction-limit-cp','250','--initial-checkpoint','ml/artifacts/quiet-ranking-v8.pt',
            '--learning-rate','.0003','--rank-weight','2','--rank-margin-cp','20',
            '--hard-pair-weight','4','--selection','ranking','--epochs','30','--patience','6'])
        run('heldout-ranking',['-m','ml.evaluate_blends','--data',DATA,'--checkpoint',CP,
            '--previous','ml/artifacts/quiet-ranking-v8.pt','--output',str(ART/'blends.json')])
        games={}
        # Predeclared ablation: same correction weight, gate, starts, and clock.
        for name,checkpoint,extra in [
            ('v8-reference','ml/artifacts/quiet-ranking-v8.pt',['--reference-inference']),
            ('v8-fast','ml/artifacts/quiet-ranking-v8.pt',[]),
            ('v10-trained',CP,[])]:
            output=ART/name
            run(name,['-m','ml.compare','--checkpoint',checkpoint,'--nn-weight','.25',
                '--quiet-only','--frozen-heuristic','--openings','benchmarks/openings-neural-v10.json',
                '--pairs','4','--time-ms','250','--max-plies','600','--output',str(output),*extra])
            games[name]=json.loads((output/'report.json').read_text())['summary']
        status('completed',matches=games,policy='Three fixed ablations at 250ms vs frozen 9e23b7c heuristic. '
               'Eight games each are screening evidence; no automatic promotion or Elo inference. '
               'Four additional paired opening starts remain reserved.')
    except Exception as exc:
        status('failed',error=str(exc));raise


if __name__=='__main__':main()
