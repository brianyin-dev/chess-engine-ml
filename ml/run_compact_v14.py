"""Freeze fresh starts and screen the compact model against both fixed baselines."""
import json
from pathlib import Path
import subprocess
import sys
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/compact-v14'
CHECKPOINT='ml/artifacts/compact-v14/trained.pt'
OLD='ml/artifacts/quiet-ranking-v8.pt'


def main():
    state=ART/'status.json'
    if state.exists():raise FileExistsError('Preserve prior experiment')
    def status(phase,**extra):
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps({'phase':phase,**extra},indent=2)+'\n');temp.replace(state)
        print(phase,flush=True)
    def run(phase,args):
        status(phase,command=args);subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    try:
        with exclusive_cpu('compact NN isolated screening'):
            run('tests',['-m','unittest','discover','-s','tests','-q'])
            run('freeze-openings',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-compact-v14.json',
                '--pairs','50','--seed','142500','--exclude-data','ml/data/aligned-v11-2026'])
            starts=json.loads((ROOT/'benchmarks/openings-compact-v14.json').read_text())
            for name,items in [('pilot',starts[:10]),('extension',starts[10:])]:
                (ROOT/f'benchmarks/openings-compact-v14-{name}.json').write_text(json.dumps(items,indent=2)+'\n')
            common=['-m','ml.compare','--checkpoint',CHECKPOINT,'--nn-weight','.25','--quiet-only',
                '--openings','benchmarks/openings-compact-v14-pilot.json','--pairs','10','--time-ms','250','--max-plies','1000']
            results={}
            for name,extra in [('vs-heuristic',['--frozen-heuristic']),
                               ('vs-old-NN',['--opponent-checkpoint',OLD,'--opponent-nn-weight','.25','--opponent-quiet-only'])]:
                output=ART/name;run(name,[*common,*extra,'--output',str(output)])
                results[name]=json.loads((output/'report.json').read_text())['summary']
            encouraging=(all(s['completed']==20 for s in results.values())
                          and results['vs-heuristic']['score_fraction_completed']>=.6
                          and results['vs-old-NN']['score_fraction_completed']>.5)
            if encouraging:
                output=ART/'vs-heuristic-extension'
                run('extend-heuristic-to-100',['-m','ml.compare','--checkpoint',CHECKPOINT,'--nn-weight','.25','--quiet-only',
                    '--frozen-heuristic','--openings','benchmarks/openings-compact-v14-extension.json','--pairs','40',
                    '--time-ms','250','--max-plies','1000','--output',str(output)])
                initial=json.loads((ART/'vs-heuristic/report.json').read_text())['games']
                extension=json.loads((output/'report.json').read_text())['games']
                for game in extension:game['pair']+=10
                results['vs-heuristic-100']=summarize(initial+extension)
            status('completed',results=results,extended_to_100=encouraging,
                   policy='Same fresh paired starts for both baseline screens, 250ms. Extend only for >=60% heuristic and >50% old NN, all games complete. No automatic app promotion or external games.')
    except Exception as exc:
        status('failed',error=str(exc));raise


if __name__=='__main__':main()
