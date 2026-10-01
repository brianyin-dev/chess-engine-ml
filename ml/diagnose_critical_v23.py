"""Confirm first competitive mistakes, compare models, test bounded teacher intervention."""
import json,math,random
import chess,chess.engine
from engine.evaluation import evaluate_position
from engine.search import search,_Search,INF
from benchmarks.analyze import loss_data,score_data
from benchmarks.cpu_lock import exclusive_cpu
from ml.evaluator import NeuralEvaluator
from ml.dataset import correction_factor
from ml.run_search_v22 import ROOT,OLD,CP as PREVIOUS,exact,identity
from ml.run_balanced_v19 import write,digest
ART=ROOT/'ml/artifacts/critical-v23'
REPORT=ROOT/'ml/artifacts/search-smooth-v22/vs-heuristic/report.json'
SF=ROOT/'tools/stockfish-sf19/stockfish/stockfish-macos-universal'


def critical(best,played):
    damage=loss_data(best,played)
    crossing=(best['cp'] is not None and best['cp']>-150 and played['cp'] is not None and played['cp']<=-150)
    competitive=(best['cp'] is not None and best['cp']>=-150) or (best['mate'] is not None and best['mate']>0)
    qualifies=competitive and (damage['allows_mate'] or crossing and (damage['cp_loss'] or 0)>=50 or (damage['cp_loss'] or 0)>=150 or damage['missed_forced_mate'])
    return {**damage,'crosses_minus_150':crossing,'competitive_before_move':competitive,'qualifies':qualifies}


def review(sf,b,move,nodes):
    best=exact(sf,b,nodes=nodes)
    played=best if move==best['pv'][0] else exact(sf,b,[move],nodes)
    a,z=score_data(best['score'],b.turn),score_data(played['score'],b.turn)
    return {'best_move':best['pv'][0].uci(),'best_score':a,'played_score':z,'best_depth':best.get('depth'),'played_depth':played.get('depth'),'nodes_per_search':nodes,**critical(a,z)}


class Trace(_Search):
    """Diagnostic full-window alpha-beta returning a score-bearing quiet leaf."""
    def qtrace(self,b,alpha,beta,ply):
        self.visit(ply,quiescence=True);check=b.is_check()
        moves=list(b.legal_moves) if check else None
        terminal=self.terminal(b,moves if check else next(iter(b.legal_moves),None),ply)
        if terminal is not None:return terminal,None
        chosen=None
        if not check:
            stand=self.static(b);leaf={'fen':b.fen(),'white_cp':stand if b.turn else -stand,'history':[m.uci() for m in b.move_stack]}
            if stand>=beta:return stand,leaf
            if stand>alpha:alpha=stand;chosen=leaf
            from engine.search import _tactical_moves
            moves=_tactical_moves(b)
        for move in self.ordered(b,moves,ply=ply):
            with self.pushed(b,move):v,leaf=self.qtrace(b,-beta,-alpha,ply+1);v=-v
            if v>=beta:return v,leaf
            if v>alpha:alpha=v;chosen=leaf
        return alpha,chosen
    def ntrace(self,b,depth,alpha,beta,ply):
        if depth<=0:return self.qtrace(b,alpha,beta,ply)
        self.visit(ply);moves=list(b.legal_moves);terminal=self.terminal(b,moves,ply)
        if terminal is not None:return terminal,None
        best=-INF;chosen=None
        for move in self.ordered(b,moves,ply=ply):
            with self.pushed(b,move):v,leaf=self.ntrace(b,depth-1,-beta,-alpha,ply+1);v=-v
            if v>best:best=v;chosen=leaf
            alpha=max(alpha,v)
            if alpha>=beta:break
        return best,chosen


def forced(b,move,model,depth=3):
    worker=Trace(b,model,None,False)
    with worker.pushed(b,move):score,leaf=worker.ntrace(b,depth-1,-INF,INF,1)
    return {'root_score_cp':-score,'leaf':leaf,'depth':depth,'nodes':worker.nodes,'qnodes':worker.qnodes}


class BoundedTeacher:
    cacheable_by_fen=True
    def __init__(self,sf):self.sf=sf;self.rows={};self.mates=0
    def evaluate_position(self,b):
        base=evaluate_position(b);factor=correction_factor(b,.25,True)
        if not factor:return base
        fen=b.fen()
        if fen not in self.rows:
            info=exact(self.sf,b,nodes=8000);cp=info['score'].white().score()
            self.rows[fen]={'score_cp':cp,'depth':info.get('depth')}
        cp=self.rows[fen]['score_cp']
        if cp is None:self.mates+=1;return base
        # Same quiet25%,250cp residual bound and whole-endpoint rounding asNN.
        limit=250*factor
        return round(base+max(-limit,min(limit,cp-base)))
    __call__=evaluate_position


def main():
    if ART.exists():raise FileExistsError('Preserve critical analysis')
    ART.mkdir();protocol={'threshold':'Scores fromourmover POV. First confirmed competitive mistake/loss: crossing>-150to<=-150 with>=50cp drop, or>=150cp regret, allowsmate ormissedforcedmate; bestcontinuation must>=-150(orpositive mate). -150 is riskflag, notproof oflostgame. Already losingpositions areseparate,notnewmistakes.','teacher':'16knode screenallcurrentmoves untilfirstqualified; confirm256knodes same-rootbest andforcedplayed, last-exactPV; fullgamehistory.','probes':'Allthree exactdepth3, then250ms/depth8; alternate evaluator order bycase beforeteacher scoring256k. Fixeddepth4 followupdiagnoses horizons; notpurecausal proof.','bounded':'ForceHCE better/playedrootmoves atdepth3 withtracedquietscore-bearing leaves. Globalroot gap>2ceil62.5=126 cannotbereversed byanyboundedquiet25% evaluator onthe same complete fixed-depth minimax tree. Gap<=126onlypermits,notproves. Test actual8knode boundedteacher leafintervention, full depth3, thenconfirmitsmove256k. Teacher oracle isdiagnostic,notbenchmark/app evaluator.','split':'Openingpair owns bothcolors; deterministic familyhash60/20/20 inherited. Trainingonlytraincriticalcases; unseenval/test decisionsheldout. Legacyretentioncanonical aliasesfiltered. No gamepilot unlessnewNNimprovesunseen searcheddecisions.'}
    write(ART/'protocol.json',protocol)
    models={'heuristic':NeuralEvaluator(OLD,0,True,incremental=True,fast_features=True),'old25':NeuralEvaluator(OLD,.25,True,incremental=True,fast_features=True),'new25':NeuralEvaluator(PREVIOUS,.25,True,incremental=True,fast_features=True)}
    diagnostics=[]
    with exclusive_cpu('v23 critical loss analysis and bounded intervention'):
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            for gi,g in enumerate(json.loads(REPORT.read_text())['games']):
                if g['current_result']!='loss':continue
                b=chess.Board(g['initial_fen'])
                for u in g['opening_moves']:b.push_uci(u)
                gid,split=identity(g['initial_fen'],g['opening_moves']);failure=None;already_losing=0
                for ply,item in enumerate(g['moves'],1):
                    move=chess.Move.from_uci(item['uci'])
                    if item['engine']=='current':
                        check=review(sf,b,move,16000)
                        if check['best_score']['cp'] is not None and check['best_score']['cp']<-150:already_losing+=1
                        if check['qualifies']:
                            check=review(sf,b,move,256000)
                            if check['qualifies']:
                                failure={'fen':b.fen(),'initial_fen':g['initial_fen'],'history':[m.uci() for m in b.move_stack],'played':move.uci(),'ply':ply,'review':check,'probes':{}}
                                names=list(models);names=names[len(diagnostics)%3:]+names[:len(diagnostics)%3]
                                # Do all searches before teacher work for eachmode.
                                for mode,options in [('depth3',{'depth':3}),('250ms',{'depth':8,'time_limit':.25}),('depth4',{'depth':4})]:
                                    probes={}
                                    for name in names:
                                        p=search(b.copy(stack=True),eval_fn=models[name],**options)
                                        if mode!='250ms':assert p.depth==options['depth'] or abs(p.score or 0)>28000
                                        probes[name]={'move':p.move.uci(),'depth':p.depth,'score':p.score,'nodes':p.nodes,'qnodes':p.qnodes,'elapsed':p.elapsed}
                                    cache={}
                                    for name,p in probes.items():
                                        cache.setdefault(p['move'],None)
                                        if cache[p['move']] is None:cache[p['move']]=review(sf,b,chess.Move.from_uci(p['move']),256000)
                                        p['review']=cache[p['move']]
                                    failure['probes'][mode]=probes
                                bestmove=chess.Move.from_uci(check['best_move']);traces={}
                                for label,mv in [('played',move),('teacher_best',bestmove),('heuristic_choice',chess.Move.from_uci(failure['probes']['depth3']['heuristic']['move'])),('new25_choice',chess.Move.from_uci(failure['probes']['depth3']['new25']['move']))]:traces[label]=forced(b.copy(stack=True),mv,models['heuristic'])
                                failure['forced_depth3']=traces
                                gap=traces['heuristic_choice']['root_score_cp']-traces['teacher_best']['root_score_cp']
                                failure['hce_preference_gap_cp']=gap;failure['bounded_range_permits_reversal']=gap<=126
                                if gap<=126:
                                    oracle=BoundedTeacher(sf);p=search(b.copy(stack=True),depth=3,eval_fn=oracle)
                                    oracle_review=review(sf,b,p.move,256000)
                                    failure['bounded_teacher']={'move':p.move.uci(),'score':p.score,'depth':p.depth,'quiet_labels':len(oracle.rows),'mate_label_fallbacks':oracle.mates,'review':oracle_review,'cache':oracle.rows}
                                break
                    b.push(move)
                diagnostics.append({'source_game':gi,'pair':g['pair'],'opening':g['opening'],'game_id':gid,'split':split,'already_losing_moves_screened':already_losing,'failure':failure})
                write(ART/'diagnostics.json',{'source_sha256':digest(REPORT),'stockfish_sha256':digest(SF),'games':diagnostics,'protocol':protocol})
                print('Critical loss',len(diagnostics),'split',split,'move',failure['played'] if failure else None,'gap',failure.get('hce_preference_gap_cp') if failure else None,'boundedteacher',failure.get('bounded_teacher',{}).get('move') if failure else None,flush=True)
    write(ART/'status.json',{'phase':'diagnosed','losses':len(diagnostics),'confirmed':sum(r['failure'] is not None for r in diagnostics)})
if __name__=='__main__':main()
