"""Frozen single-feature diagnostic with independent searched-choice gate."""
import json,hashlib
import chess,chess.engine
from dataclasses import asdict
from benchmarks.diagnose_bishop_v25 import ROOT,ART,Evaluator,components,target
from benchmarks.cpu_lock import exclusive_cpu
from engine.search import search
from ml.train_critical_v23 import board
from ml.diagnose_critical_v23 import SF,review
from ml.confirm_fatal_v23 import fatal

def save(name,value):(ART/name).write_text(json.dumps(value,indent=2)+'\n')

def probe(sf,r,mode,models):
    opts={'depth':int(mode[5:])} if mode.startswith('depth') else {'depth':64,'time_limit':float(mode)}
    ps={};cache={}
    for n,e in models:
        p=search(board(r),eval_fn=e,use_lmr=True,**opts)
        ps[n]={**asdict(p),'move':p.move.uci()}
    for n,p in ps.items():
        if p['move'] not in cache:cache[p['move']]=review(sf,board(r),chess.Move.from_uci(p['move']),256000)
        p['review']=cache[p['move']]
    return ps

def main():
    ART.mkdir(parents=True,exist_ok=False)
    save('protocol.json',{'penalty_cp':220,'feature':'Advanced queens only, <=1 geometrically unattacked destination. No parameter sweep. Root bishop position is development evidence only.',
        'gate':'On32priorheldout roots, candidate meanregret no worse thanbaseline, noadditional>=150cp moves oravoidablelosingtransitions, exactdepth3 andisolated750ms. Target750ms regret improves by>=50cp. No games unlessgatepasses.',
        'baseline_evaluation_sha256':hashlib.sha256((ROOT/'benchmarks/evaluation_baseline_v25.py').read_bytes()).hexdigest(),
        'search':'Unchanged v24 LMR, same heuristic except one explicit feature. No NN.'})
    models=[('baseline',Evaluator()),('candidate',Evaluator(220))]
    with exclusive_cpu('v25 bishop evaluation diagnosis and heldout searched choices'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32})
        f=target();details=[]
        for name,fen in [('root',f['fen'])]+[(n,t['leaf']['fen']) for n,t in f['forced'].items() if t['leaf']]:
            b=chess.Board(fen);info=sf.analyse(b,chess.engine.Limit(nodes=2000000),game=object());v=b.copy();line=[]
            for m in info.get('pv',[])[:20]:line.append(v.san(m));v.push(m)
            details.append({'name':name,'fen':fen,'components':components(b),'baseline_white_cp':models[0][1](b),'candidate_white_cp':models[1][1](b),'stockfish_white_cp':info['score'].white().score(),'stockfish_pv_san':line})
        save('component-diagnosis.json',details)
        tp={mode:probe(sf,f,mode,models) for mode in ('depth2','depth3','depth4','0.25','0.75','2.0')}
        save('target-probes.json',tp)
        cases=json.loads((ROOT/'ml/artifacts/critical-v23/test-roots.json').read_text());records=[]
        for i,r in enumerate(cases):
            order=models if i%2==0 else list(reversed(models))
            ps={mode:probe(sf,r,mode,order) for mode in ('depth3','0.75')}
            records.append({'root':r,'probes':ps});save('heldout.json',records)
            print('Heldout screen',len(records),'/32',flush=True)
        summary={}
        for mode in ('depth3','0.75'):
            summary[mode]={}
            for n,_ in models:
                ps=[r['probes'][mode][n] for r in records];losses=[p['review']['cp_loss'] for p in ps if p['review']['cp_loss'] is not None]
                summary[mode][n]={'positions':len(ps),'comparable':len(losses),'mean_cp_regret':sum(losses)/len(losses),'bad_150cp_moves':sum((p['review']['cp_loss'] or 0)>=150 for p in ps),'losing_transitions':sum(fatal(p['review']) for p in ps)}
        save('heldout-summary.json',summary)
        improvement=tp['0.75']['baseline']['review']['cp_loss']-tp['0.75']['candidate']['review']['cp_loss']
        checks=[]
        for mode,ps in summary.items():
            a,z=ps['baseline'],ps['candidate'];checks.append(z['comparable']>=24 and z['mean_cp_regret']<=a['mean_cp_regret'] and z['bad_150cp_moves']<=a['bad_150cp_moves'] and z['losing_transitions']<=a['losing_transitions'])
        save('decision.json',{'target_750ms_regret_improvement_cp':improvement,'heldout_gates':checks,'qualifies_for_games':all(checks) and improvement>=50,'app_promoted':False})

if __name__=='__main__':main()
