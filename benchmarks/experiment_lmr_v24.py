"""Frozen baseline, three-budget mistake screen and fresh paired LMR games."""
import json,hashlib
from dataclasses import asdict
from pathlib import Path
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.search_baseline_v24 import search as baseline
from engine.search import search
from benchmarks.match import play_game,summarize
from ml.train_critical_v23 import board
from ml.diagnose_critical_v23 import SF,review
from ml.confirm_fatal_v23 import fatal

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'benchmarks/results/lmr-v24'

def save(name,data):
    (ART/name).write_text(json.dumps(data,indent=2)+'\n')

def main():
    ART.mkdir(parents=True,exist_ok=False)
    save('protocol.json',{'budgets_seconds':[.25,.75,2.], 'depth_cap':64,
        'evaluation':'Unchanged app heuristic, no NN or book after prescribed start.',
        'lmr':'One-ply reduction, fifth or later move, depth>=3, non-root quiet moves only; exclude check evasions, giving check, preferred/killer/history moves. Full-depth verification for every reduced score above alpha.',
        'decision':'Keep opt-in only unless fewer critical mistakes on known positions AND >50% score over20 fresh paired games; small pilot cannot establish general strength or Elo.',
        'baseline_sha256':hashlib.sha256((ROOT/'benchmarks/search_baseline_v24.py').read_bytes()).hexdigest()})
    with exclusive_cpu('v24 three-budget screen then fresh paired search games'):
        cases=[r['failure'] for r in json.loads((ROOT/'ml/artifacts/critical-v23/fatal-and-reachable.json').read_text())['games'] if r['failure']]
        records=[]
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            for i,r in enumerate(cases):
                probes={};cache={}
                for budget in (.25,.75,2.):
                    order=['baseline','lmr'] if i%2==0 else ['lmr','baseline']
                    ps={}
                    for name in order:
                        fn=baseline if name=='baseline' else search
                        opts={} if name=='baseline' else {'use_lmr':True}
                        result=fn(board(r),depth=64,time_limit=budget,**opts)
                        ps[name]={**asdict(result),'move':result.move.uci()}
                    for name,p in ps.items():
                        if p['move'] not in cache:cache[p['move']]=review(sf,board(r),chess.Move.from_uci(p['move']),256000)
                        p['review']=cache[p['move']]
                    probes[str(budget)]=ps
                records.append({'root':r,'probes':probes})
                save('mistakes.json',records);print('Mistake screen',len(records),'/11',flush=True)
        summary={}
        for budget in (.25,.75,2.):
            summary[str(budget)]={}
            for name in ('baseline','lmr'):
                ps=[r['probes'][str(budget)][name] for r in records]
                losses=[p['review']['cp_loss'] for p in ps if p['review']['cp_loss'] is not None]
                summary[str(budget)][name]={'mean_depth':sum(p['depth'] for p in ps)/len(ps),'mean_cp_regret':sum(losses)/len(losses),'avoidable_losing_transitions':sum(fatal(p['review']) for p in ps),'bad_150cp_moves':sum((p['review']['cp_loss'] or 0)>=150 for p in ps)}
        save('mistake-summary.json',summary)
        openings=json.loads((ROOT/'benchmarks/openings-lmr-v24.json').read_text())
        games=[];pgns=[]
        def selector(name,b,time_limit,depth_cap):
            fn=search if name=='current' else baseline
            opts={'use_lmr':True} if name=='current' else {}
            p=fn(b,depth=depth_cap,time_limit=time_limit,**opts)
            stats=asdict(p);stats.pop('move');stats.update(budget_seconds=time_limit,overrun_seconds=max(0,p.elapsed-time_limit))
            return p.move,stats
        for i,opening in enumerate(openings,1):
            for color in (chess.WHITE,chess.BLACK):
                record,pgn=play_game(opening,color,.25,400,64,selector,i,'baseline')
                games.append(record);pgns.append(pgn)
                save('report.json',{'games':games,'summary':summarize(games),'status':'running' if len(games)<20 else 'completed'})
                (ART/'games.pgn').write_text('\n'.join(pgns))
                print('Game',len(games),'/20',record['current_result'],summarize(games),flush=True)
        s=summarize(games)
        screen_better=sum(x['lmr']['avoidable_losing_transitions'] for x in summary.values())<sum(x['baseline']['avoidable_losing_transitions'] for x in summary.values())
        game_better=s['completed']==20 and s['score_fraction_completed']>.5 and not s['errors']
        save('decision.json',{'screen_better':screen_better,'game_better':game_better,'qualifies_for_default':screen_better and game_better,'summary':s,'promotion_requires_review_of_results':True})

if __name__=='__main__':main()
