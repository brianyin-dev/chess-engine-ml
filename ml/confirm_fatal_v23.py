"""First -150 crossings and bounded corrections at the depth actually reached."""
import json
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from engine.evaluation import evaluate_position
from engine.search import search
from ml.dataset import correction_factor
from ml.diagnose_critical_v23 import ART,REPORT,SF,OLD,PREVIOUS,review,forced
from ml.run_search_v22 import exact,identity
from ml.run_balanced_v19 import write
from ml.evaluator import NeuralEvaluator


def fatal(review):
    a,z=review['best_score'],review['played_score']
    avoidable=(a['cp'] is not None and a['cp']>-150) or (a['mate'] is not None and a['mate']>0)
    losing=(z['cp'] is not None and z['cp']<=-150) or (z['mate'] is not None and z['mate']<0)
    substantial=(review['cp_loss'] or 0)>=50 or review['allows_mate'] or review['missed_forced_mate']
    return avoidable and losing and substantial


class StrongBoundedTeacher:
    cacheable_by_fen=True
    def __init__(self,sf):self.sf=sf;self.rows={}
    def evaluate_position(self,b):
        base=evaluate_position(b);f=correction_factor(b,.25,True)
        if not f:return base
        fen=b.fen()
        if fen not in self.rows:
            i=exact(self.sf,b,nodes=64000);self.rows[fen]={'score_cp':i['score'].white().score(),'mate':i['score'].white().mate(),'depth':i.get('depth')}
        row=self.rows[fen];limit=250*f
        delta=(limit if row['mate']>0 else -limit) if row['score_cp'] is None else max(-limit,min(limit,row['score_cp']-base))
        return round(base+delta)
    __call__=evaluate_position


def main():
    if (ART/'fatal-and-reachable.json').exists():raise FileExistsError('Preserve fatal analysis')
    write(ART/'fatal-protocol.json',{'reason':'User prioritizes actual -150 crossings. Earlier diagnostics also found significant positive-score drops. Keep both, scan firstconfirmedavoidable losing-state transition separately.','depth':'Use matched completed depth from the ORIGINAL played move,clamp1..4. Compare three evaluators atsame complete depth, thenreuse depth3/250ms/4 ifsame root. This matters because most250ms mistakes onlycompleted1or2, where a boundedNN could compensate forlimitedsearch.','oracle':'64knode teacher at everyactualquietstaticcall, same25% gate/250cp bound/rounding; mate labels map todirection ofmaxallowedshift,notinfinite NN values. Force rootteacherbest/baselinechoice withscore-bearing tracedleaf. Confirm resulting rootmove256knodes. Root gap>126rulesoutreversal onthatfixedtree; smaller gap alone isnot enough.','training_eligibility':'Onlytrainingfamily cases where boundedteacher improves overcurrentNN atmatcheddepth by>=50cp andavoids the losing-stateflag, or removesnewlyallowedmate. Require canonicalnewquietlabels andteacherconfirmation ofleafpair. Nooracle timings arestrength claims.'})
    initial=json.loads((ART/'diagnostics.json').read_text())['games'];games=json.loads(REPORT.read_text())['games'];out=[]
    models={'heuristic':NeuralEvaluator(OLD,0,True,incremental=True,fast_features=True),'old25':NeuralEvaluator(OLD,.25,True,incremental=True,fast_features=True),'new25':NeuralEvaluator(PREVIOUS,.25,True,incremental=True,fast_features=True)}
    with exclusive_cpu('v23 fatal threshold and matched-depth teacher intervention'):
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            for original in initial:
                g=games[original['source_game']];b=chess.Board(g['initial_fen'])
                for u in g['opening_moves']:b.push_uci(u)
                failure=None
                for ply,item in enumerate(g['moves'],1):
                    move=chess.Move.from_uci(item['uci'])
                    if item['engine']=='current':
                        first=original['failure']
                        oldcheck=first['review'] if first and b.fen()==first['fen'] else None
                        check=oldcheck or review(sf,b,move,16000)
                        if fatal(check):
                            check=oldcheck or review(sf,b,move,256000)
                            if fatal(check):
                                failure={'fen':b.fen(),'initial_fen':g['initial_fen'],'history':[m.uci() for m in b.move_stack],'played':move.uci(),'original_played_depth':item['depth'],'ply':ply,'review':check}
                                break
                    b.push(move)
                row={k:original[k] for k in ('source_game','pair','opening','game_id','split')};row['failure']=failure
                if failure:
                    depth=max(1,min(4,failure['original_played_depth']));probes={}
                    for name,model in models.items():
                        p=search(b.copy(stack=True),depth=depth,eval_fn=model)
                        assert p.depth==depth or abs(p.score or 0)>28000
                        probes[name]={'move':p.move.uci(),'depth':p.depth,'score':p.score,'nodes':p.nodes,'qnodes':p.qnodes}
                    cached={}
                    for name,p in probes.items():
                        if p['move'] not in cached:cached[p['move']]=review(sf,b,chess.Move.from_uci(p['move']),256000)
                        p['review']=cached[p['move']]
                    failure['matched_depth']=depth;failure['probes']=probes
                    teacher=chess.Move.from_uci(check['best_move']);hce=chess.Move.from_uci(probes['heuristic']['move'])
                    traces={}
                    for name,mv in [('teacher_best',teacher),('heuristic_choice',hce),('played',move),('new25_choice',chess.Move.from_uci(probes['new25']['move']))]:traces[name]=forced(b.copy(stack=True),mv,models['heuristic'],depth)
                    failure['forced']=traces;gap=traces['heuristic_choice']['root_score_cp']-traces['teacher_best']['root_score_cp'];failure['hce_gap_cp']=gap
                    failure['range_permits_reversal']=gap<=126
                    if gap<=126:
                        oracle=StrongBoundedTeacher(sf);p=search(b.copy(stack=True),depth=depth,eval_fn=oracle)
                        r=review(sf,b,p.move,256000)
                        prev=probes['new25']['review'];improvement=(prev['cp_loss']-r['cp_loss']) if prev['cp_loss'] is not None and r['cp_loss'] is not None else None
                        avoids=(r['played_score']['cp'] is not None and r['played_score']['cp']>-150) or (r['played_score']['mate'] is not None and r['played_score']['mate']>0)
                        repair=(improvement is not None and improvement>=50 and avoids) or prev['allows_mate'] and not r['allows_mate']
                        failure['oracle']={'move':p.move.uci(),'depth':p.depth,'score':p.score,'quiet_labels':len(oracle.rows),'review':r,'regret_improvement_cp':improvement,'avoids_losing_flag':avoids,'demonstrated_repair':repair,'cache':oracle.rows}
                        if repair:failure['forced']['oracle_choice']=forced(b.copy(stack=True),p.move,models['heuristic'],depth)
                out.append(row);write(ART/'fatal-and-reachable.json',{'games':out})
                print('Fatal case',len(out),'split',row['split'],'depth',failure['matched_depth'] if failure else None,'gap',failure.get('hce_gap_cp') if failure else None,'repair',failure.get('oracle',{}).get('demonstrated_repair') if failure else None,flush=True)
    write(ART/'status.json',{'phase':'fatal-diagnosed','losses':len(out),'fatal_crossings':sum(r['failure'] is not None for r in out),'demonstrated_bounded_repairs':sum(bool(r['failure'] and r['failure'].get('oracle',{}).get('demonstrated_repair')) for r in out)})
if __name__=='__main__':main()
