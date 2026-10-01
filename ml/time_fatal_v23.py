"""Complete truefatal root controls: exactdepth3, then isolated250ms."""
import json
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from engine.search import search
from ml.train_critical_v23 import board,baseline_models
from ml.diagnose_critical_v23 import ART,SF,review
from ml.run_balanced_v19 import write

def main():
    if (ART/'fatal-model-controls.json').exists():raise FileExistsError('Preserve controls')
    rows=json.loads((ART/'fatal-and-reachable.json').read_text())['games'];models=baseline_models();result=[]
    with exclusive_cpu('v23 truefatal exactdepth and equaltime controls'):
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            for i,row in enumerate(rows):
                f=row['failure']
                if not f:continue
                names=list(models);names=names[i%3:]+names[:i%3];probes={}
                for mode,opts in [('depth3',{'depth':3}),('250ms',{'depth':8,'time_limit':.25})]:
                    ps={}
                    for n in names:
                        p=search(board(f),eval_fn=models[n],**opts)
                        ps[n]={'move':p.move.uci(),'depth':p.depth,'score':p.score,'nodes':p.nodes,'qnodes':p.qnodes,'elapsed':p.elapsed}
                    cache={}
                    for n,p in ps.items():
                        if p['move'] not in cache:cache[p['move']]=review(sf,board(f),chess.Move.from_uci(p['move']),256000)
                        p['review']=cache[p['move']]
                    probes[mode]=ps
                result.append({'source_game':row['source_game'],'split':row['split'],'fen':f['fen'],'probes':probes});print('Fatal model controls',len(result),flush=True)
    write(ART/'fatal-model-controls.json',{'games':result,'policy':'Truefirstconfirmedlosing-statecrossings. Allcontrolsoriginalheuristic/v8/v22. Exactdepth3 first,250ms/depth8 second, orderrotatedbygame. Teacher256knodes afterallsearches ineachmode. HeldexclusiveCPU, no concurrenttraining/benchmark.'})
if __name__=='__main__':main()
