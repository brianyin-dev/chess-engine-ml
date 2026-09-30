"""Isolated verification, deployed-blend training, and comparisons with both baselines."""
import json
from pathlib import Path
import subprocess
import sys
import time
from contextlib import nullcontext
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/aligned-v11'
SF='tools/stockfish-sf19/stockfish/stockfish-macos-universal'
OLD='ml/artifacts/quiet-ranking-v8.pt'
BASE='ml/data/targeted-v10-2026'
NEW='ml/data/conversion-v11-2026'
DATA='ml/data/aligned-v11-2026'
CP='ml/artifacts/aligned-ranking-v11.pt'


def main():
    ART.mkdir(parents=True,exist_ok=True)
    if (ART/'status.json').exists() and json.loads((ART/'status.json').read_text()).get('phase') not in (
            'waiting-for-other-benchmarks', 'waiting-for-old-NN-20-game-verification'):
        raise FileExistsError('Preserve existing experiment status')
    def status(phase,**extra):
        temp=ART/'status.tmp'; temp.write_text(json.dumps({'phase':phase,**extra},indent=2)+'\n')
        temp.replace(ART/'status.json');print(phase,flush=True)
    def idle():
        while True:
            active=[]
            for path in (ROOT/'benchmarks/results').glob('**/report.json'):
                try:r=json.loads(path.read_text())
                except ValueError:
                    active.append(str(path));continue
                except OSError:continue
                if r.get('status')=='running' and not r.get('finished_at'):active.append(str(path))
            if not active:return
            status('waiting-for-other-benchmarks',reports=active);time.sleep(5)
    def run(phase,args):
        idle(); status(phase,command=args)
        # Match commands acquire the lock themselves; other CPU work needs
        # the parent to hold it so a concurrently launched match waits.
        lock = nullcontext() if args[:2] == ['-m','ml.compare'] else exclusive_cpu(phase)
        with lock:
            subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    pilot=ART/'old-v8-verified-20/report.json'
    try:
        run('tests',['-m','unittest','discover','-s','tests','-q'])
        run('isolated-old-NN-20-game-verification',['-m','ml.compare','--checkpoint',OLD,
            '--nn-weight','.25','--quiet-only','--frozen-heuristic',
            '--openings','benchmarks/openings-neural-v11-pilot.json','--pairs','10',
            '--time-ms','250','--max-plies','1000','--output',str(pilot.parent)])
        initial=json.loads(pilot.read_text())
        encouraging=(initial['summary']['completed']==20 and initial['summary']['score_fraction_completed']>=.6)
        verification=initial['summary']
        if encouraging:
            output=ART/'old-v8-isolated-extension-80'
            run('extend-old-NN-to-100',['-m','ml.compare','--checkpoint',OLD,'--nn-weight','.25',
                '--quiet-only','--frozen-heuristic','--openings','benchmarks/openings-neural-v11-extension.json',
                '--pairs','40','--time-ms','250','--max-plies','1000','--output',str(output)])
            extension=json.loads((output/'report.json').read_text())
            for game in extension['games']:game['pair']+=10
            verification=summarize(initial['games']+extension['games'])
        (ART/'verification-summary.json').write_text(json.dumps({'extension_threshold':'.6 score with all 20 games complete',
            'extended_to_100':encouraging,'summary':verification},indent=2)+'\n')
        run('isolated-inference-profile',['-m','ml.profile_leaf_inference','--checkpoint',OLD,
            '--data',BASE+'/test.jsonl','--output',str(ART/'isolated-inference-profile.json')])
        run('review-losses-and-draws',['-m','ml.analyze_losses','--report',str(pilot),
            '--report','ml/artifacts/targeted-v10/v10-trained/report.json',
            '--checkpoint',OLD,'--nn-weight','.25','--quiet-only','--include-draws',
            '--stockfish',SF,'--output',str(ART/'losses-and-draws.json')])
        run('conversion-training-examples',['-m','ml.generate_quiet_rankings','--data',BASE,
            '--checkpoint',OLD,'--failures',str(ART/'losses-and-draws.json'),'--stockfish',SF,
            '--output',NEW,'--train-pairs','600','--heldout-pairs','100','--failure-id-offset','3000000'])
        run('retain-existing-supervision',['-m','ml.merge_rankings','--old',BASE,'--new',NEW,'--output',DATA])
        run('aligned-training',['-m','ml.train','--data',DATA,'--pairs',DATA+'/pairs',
            '--checkpoint',CP,'--metrics','ml/artifacts/aligned-ranking-v11.json',
            '--features','relationships','--target','residual','--color-consistent','--correction-limit-cp','250',
            '--correction-weight','.25','--quiet-only','--initial-checkpoint',OLD,
            '--learning-rate','.0003','--rank-weight','2','--rank-margin-cp','10',
            '--hard-pair-weight','4','--selection','ranking','--epochs','30','--patience','6'])
        run('aligned-validation',['-m','ml.evaluate_blends','--data',DATA,'--checkpoint',CP,
            '--previous',OLD,'--output',str(ART/'validation.json')])
        run('freeze-fresh-comparison-openings',['-m','benchmarks.generate_strength_openings',
            '--output','benchmarks/openings-neural-v11-comparison.json','--pairs','10','--seed','112500','--exclude-data',DATA])
        common=['-m','ml.compare','--checkpoint',CP,'--nn-weight','.25','--quiet-only',
            '--openings','benchmarks/openings-neural-v11-comparison.json','--pairs','10',
            '--time-ms','250','--max-plies','1000']
        games={}
        for name,extra in [('aligned-vs-heuristic',['--frozen-heuristic']),
                           ('aligned-vs-old-NN',['--opponent-checkpoint',OLD,
                              '--opponent-nn-weight','.25','--opponent-quiet-only'])]:
            output=ART/name
            run(name,[*common,*extra,'--output',str(output)])
            games[name]=json.loads((output/'report.json').read_text())['summary']
        eligible=all(s['completed']==20 and s['score_fraction_completed']>.5 for s in games.values())
        if not eligible:
            status('completed',verification=verification,matches=games,
                   candidate_eligible_for_larger_validation=False,
                   stockfish_skipped='Candidate must beat both baselines before external benchmarking.')
            return
        run('freeze-1500-openings',['-m','benchmarks.generate_strength_openings',
            '--output','benchmarks/openings-neural-v11-1500.json','--pairs','10','--seed','111501','--exclude-data',DATA])
        for name,checkpoint,weight in [('heuristic-vs-1500',OLD,'0'),('aligned-vs-1500',CP,'.25')]:
            output=ART/name
            run(name,['-m','ml.compare','--checkpoint',checkpoint,'--nn-weight',weight,'--quiet-only',
                '--openings','benchmarks/openings-neural-v11-1500.json','--pairs','10','--time-ms','250',
                '--max-plies','1000','--stockfish',SF,'--stockfish-elo','1500','--output',str(output)])
            games[name]=json.loads((output/'report.json').read_text())['summary']
        status('completed',verification=verification,matches=games,
               candidate_eligible_for_larger_validation=eligible,
               policy='250ms equal clocks. Old NN extended only for >=60% complete 20-game pilot. '
                      'Fresh model comparison starts never used to train; compare with both fixed baselines. '
                      '1500 is a configured Stockfish setting, not human Elo. No automatic app promotion.')
    except Exception as exc:
        status('failed',error=str(exc));raise


if __name__=='__main__':main()
