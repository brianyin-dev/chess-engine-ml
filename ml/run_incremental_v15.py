"""Screen unchanged v8 weights with faster feature updates, under equal clocks."""
import json
from pathlib import Path
import subprocess
import sys
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/incremental-v15'
CP='ml/artifacts/quiet-ranking-v8.pt'


def main():
    state=ART/'status.json'
    if state.exists():raise FileExistsError('Preserve existing results')
    def status(phase,**extra):
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps({'phase':phase,**extra},indent=2)+'\n');temp.replace(state)
        print(phase,flush=True)
    def run(phase,args):
        status(phase,command=args);subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    try:
        with exclusive_cpu('incremental NN isolated screening'):
            run('tests',['-m','unittest','discover','-s','tests','-q'])
            run('freeze-openings',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-incremental-v15.json',
                '--pairs','50','--seed','152500','--exclude-data','ml/data/aligned-v11-2026'])
            starts=json.loads((ROOT/'benchmarks/openings-incremental-v15.json').read_text())
            for name,items in [('pilot',starts[:10]),('extension',starts[10:])]:
                (ROOT/f'benchmarks/openings-incremental-v15-{name}.json').write_text(json.dumps(items,indent=2)+'\n')
            common=['-m','ml.compare','--checkpoint',CP,'--nn-weight','.25','--quiet-only','--incremental',
                '--openings','benchmarks/openings-incremental-v15-pilot.json','--pairs','10','--time-ms','250','--max-plies','1000']
            results={}
            for name,extra in [('vs-heuristic',['--frozen-heuristic']),
                               ('vs-reference-NN',['--opponent-checkpoint',CP,'--opponent-nn-weight','.25','--opponent-quiet-only'])]:
                output=ART/name;run(name,[*common,*extra,'--output',str(output)])
                results[name]=json.loads((output/'report.json').read_text())['summary']
            encouraging=(all(s['completed']==20 for s in results.values()) and
                results['vs-heuristic']['score_fraction_completed']>=.6 and
                results['vs-reference-NN']['score_fraction_completed']>.5)
            if encouraging:
                output=ART/'vs-heuristic-extension'
                run('extend-heuristic-to-100',['-m','ml.compare','--checkpoint',CP,'--nn-weight','.25','--quiet-only','--incremental',
                    '--frozen-heuristic','--openings','benchmarks/openings-incremental-v15-extension.json','--pairs','40',
                    '--time-ms','250','--max-plies','1000','--output',str(output)])
                initial=json.loads((ART/'vs-heuristic/report.json').read_text())['games']
                extension=json.loads((output/'report.json').read_text())['games']
                for game in extension:game['pair']+=10
                results['vs-heuristic-100']=summarize(initial+extension)
            status('completed',results=results,extended_to_100=encouraging,
                   policy='Same v8 checkpoint and .25 quiet gate for NN sides; only candidate uses incremental blocks. Fresh paired starts, 250ms. Extend only for >=60% heuristic and >50% reference NN. No app promotion or external games.')
    except Exception as exc:
        status('failed',error=str(exc));raise


if __name__=='__main__':main()
