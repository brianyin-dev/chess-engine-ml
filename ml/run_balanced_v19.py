"""Loss diagnostics, balanced settled supervision, frozen blend, fresh games."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import chess
import chess.engine
from benchmarks.analyze import review
from benchmarks.cpu_lock import exclusive_cpu
from engine.evaluation import evaluate
from engine.search import search
from ml.analyze_losses import costly
from ml.dataset import correction_factor
from ml.evaluate_blends import pair_metrics
from ml.evaluator import NeuralEvaluator
from ml.generate_search_data import key

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'ml/artifacts/balanced-v19'
NEW = ROOT/'ml/data/balanced-v19-new'
DATA = ROOT/'ml/data/balanced-v19-2026'
BASE = ROOT/'ml/data/aligned-v18-2026'
OLD = ROOT/'ml/artifacts/quiet-ranking-v8.pt'
PREVIOUS = ROOT/'ml/artifacts/disagreements-v18/trained.pt'
SF = ROOT/'tools/stockfish-sf19/stockfish/stockfish-macos-universal'
CP = ART/'trained.pt'
REPORTS = [ROOT/f'ml/artifacts/disagreements-v18/{n}/report.json' for n in ('vs-optimized-zero','vs-old-NN')]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def readrows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def models():
    return {'heuristic':NeuralEvaluator(OLD,0,True,incremental=True),
            'old':NeuralEvaluator(OLD,.25,True,incremental=True),
            'candidate-v18':NeuralEvaluator(PREVIOUS,.25,True,incremental=True)}


def diagnose(sf):
    result={'policy':'First confirmed >=100cp or mate deterioration per v18 loss. All three evaluators probed at exact depth3 with no clock/node cap, then depth8/250ms. Full game history retained. Teacher100ms screen/400ms confirmation. Fixed-depth differences diagnose evaluation/search decisions, not pure static evaluation causality.',
            'source_sha256':{str(p):digest(p) for p in REPORTS},'games':[]}
    evaluations=models()
    for source in REPORTS:
        report=json.loads(source.read_text())
        assert report['checkpoint_sha256']==digest(PREVIOUS)
        for gi,game in enumerate(report['games']):
            if game['current_result']!='loss': continue
            board=chess.Board(game['initial_fen'])
            for uci in game['opening_moves']: board.push_uci(uci)
            failure=None
            for ply,item in enumerate(game['moves'],1):
                move=chess.Move.from_uci(item['uci'])
                if item['engine']=='current':
                    screened=review(sf,board,move,.1)
                    if costly(screened):
                        confirmed=review(sf,board,move,.4)
                        if costly(confirmed):
                            failure={'fen':board.fen(),'played':move.uci(),'ply':ply,**confirmed,
                                     'history':[m.uci() for m in board.move_stack],'initial_fen':game['initial_fen'],'probes':{}}
                            for name,model in evaluations.items():
                                failure['probes'][name]={}
                                for mode,options in [('depth3',{'depth':3}),('250ms',{'depth':8,'time_limit':.25})]:
                                    probe=search(board.copy(stack=True),eval_fn=model,**options)
                                    if mode=='depth3' and probe.depth!=3: raise ValueError('Incomplete equal-depth probe')
                                    failure['probes'][name][mode]={'move':probe.move.uci(),'depth':probe.depth,
                                        'nodes':probe.nodes,'qnodes':probe.qnodes,'elapsed':probe.elapsed,
                                        'review':review(sf,board,probe.move,.4)}
                            break
                board.push(move)
            result['games'].append({'source':str(source),'game':gi,'opening':game['opening'],
                'game_id':7100000+len(result['games']),'first_costly_move':failure})
            write(ART/'diagnostics.json',result)
            print(f"Diagnosed loss {len(result['games'])}: {failure['played'] if failure else 'no confirmed error'}",flush=True)
    result['summary']={'losses':len(result['games']),'confirmed':sum(g['first_costly_move'] is not None for g in result['games']),
        'models':{}}
    for name in evaluations:
        result['summary']['models'][name]={}
        for mode in ('depth3','250ms'):
            probes=[g['first_costly_move']['probes'][name][mode] for g in result['games'] if g['first_costly_move']]
            cps=[p['review']['cp_loss'] for p in probes if p['review']['cp_loss'] is not None]
            result['summary']['models'][name][mode]={'positions':len(probes),'mean_cp_regret':sum(cps)/len(cps) if cps else None,
                'cp_comparable':len(cps),'costly_moves':sum(costly(p['review']) for p in probes),
                'mean_depth':sum(p['depth'] for p in probes)/len(probes) if probes else None,
                'mean_nodes':sum(p['nodes']+p['qnodes'] for p in probes)/len(probes) if probes else None}
    write(ART/'diagnostics.json',result)
    return result


def excluded_positions():
    seen=set()
    # All development and former holdout labels remain excluded from new cases.
    for path in (ROOT/'ml/data').glob('**/*.jsonl'):
        for row in readrows(path):
            seen.update(key(row[f]) for f in ('fen','root_fen','good_fen','bad_fen') if f in row)
    return seen


def collect(sf,diagnostics,starts_path=None,game_offset=0,append=False):
    rng=random.Random(190001); seen=excluded_positions(); evaluations=models(); old=evaluations['old']
    rows={s:{} for s in ('train','val','test')};pairs={s:[] for s in rows};owners={}
    cache={}
    if append:
        for split in rows:
            rows[split]={key(r['fen']):r for r in readrows(NEW/f'{split}.jsonl')}
            pairs[split]=readrows(NEW/'pairs'/f'{split}.jsonl')
            for r in list(rows[split].values())+pairs[split]:
                for f in ('fen','root_fen','good_fen','bad_fen'):
                    if f in r:owners[key(r[f])]=split
    def exact(board,nodes=64000):
        k=(board.fen(),nodes)
        if k in cache:return cache[k]
        last=None
        with sf.analysis(board,chess.engine.Limit(nodes=nodes),game=object()) as stream:
            for info in stream:
                if info.get('pv') and 'score' in info and not info.get('lowerbound') and not info.get('upperbound'):last=dict(info)
        if last is None:raise ValueError('No exact teacher score')
        cache[k]=last;return last
    def settle(board):
        board=board.copy(stack=False);line=[]
        for _ in range(17):
            if board.is_game_over():return None
            info=exact(board,12000);move=info['pv'][0]
            if not (board.is_check() or board.is_capture(move) or move.promotion):
                info=exact(board);cp=info['score'].white().score()
                if cp is None or abs(cp)>=1500:return None
                return board,cp,line
            line.append(move.uci());board.push(move)
        return None
    def add_root(board,split,gid,source,forced_bad=None,allow_seen_root=False):
        rootkey=key(board.fen())
        if board.is_game_over() or (rootkey in seen and not allow_seen_root) or owners.get(rootkey,split)!=split:return
        teacher=exact(board,24000);best=teacher['pv'][0]
        a=board.copy();a.push(best);good=settle(a)
        if good is None:return
        # Diverse alternatives, including old decisions regardless of correctness.
        old_move=search(board.copy(stack=True),depth=2,node_limit=500,eval_fn=old).move
        hce_move=search(board.copy(stack=True),depth=2,node_limit=500,eval_fn=evaluations['heuristic']).move
        alternatives=[forced_bad,old_move,hce_move,*rng.sample(list(board.legal_moves),min(2,board.legal_moves.count()))]
        for move in dict.fromkeys(alternatives):
            if move is None or move==best:continue
            # Confirm alternatives at the same root, before settled ranking labels.
            confirmed=review(sf,board,move,.2)
            if not costly(confirmed) and (confirmed['cp_loss'] or 0)<50:continue
            b=board.copy();b.push(move);bad=settle(b)
            if bad is None:continue
            sign=1 if board.turn else -1;gap=sign*(good[1]-bad[1])
            endpointkeys={key(good[0].fen()),key(bad[0].fen())}
            if gap<50 or len(endpointkeys)<2 or endpointkeys&seen:continue
            if any(owners.get(k,split)!=split for k in endpointkeys):continue
            if not any(correction_factor(leaf,.25,True) for leaf in (good[0],bad[0])):continue
            # Avoid assigning identical board aliases to different source games.
            if any(k in owners for k in endpointkeys):continue
            hgap=sign*(evaluate(good[0])-evaluate(bad[0]));ogap=sign*(old(good[0])-old(bad[0]))
            pair={'root_fen':board.fen(),'good_fen':good[0].fen(),'bad_fen':bad[0].fen(),'sign':sign,
                'game_id':gid,'source':source,'good_move':best.uci(),'bad_move':move.uci(),
                'good_tactical_line':good[2],'bad_tactical_line':bad[2],'cp_loss':gap,'heuristic_gap_cp':hgap,
                'old_gap_cp':ogap,'root_review':confirmed,'endpoint_nodes':64000,
                'training_weight':3 if split=='train' and ogap>0 else 2 if split=='train' and source=='confirmed_failure' else 1}
            pairs[split].append(pair);owners[rootkey]=split
            for leaf,cp in ((good[0],good[1]),(bad[0],bad[1])):
                k=key(leaf.fen());owners[k]=split
                rows[split][k]={'fen':leaf.fen(),'score_cp':cp,'game_id':gid,'ply':leaf.ply(),'source':source,'root_fen':board.fen()}
        # Also retain ordinary capture-free board score labels.
        if not board.is_check() and not next(board.generate_legal_captures(),None) and rootkey not in seen and rootkey not in rows[split]:
            info=exact(board);cp=info['score'].white().score()
            if cp is not None and abs(cp)<1500:
                rows[split][rootkey]={'fen':board.fen(),'score_cp':cp,'game_id':gid,'ply':board.ply(),'source':'ordinary_quiet'}
                owners[rootkey]=split
    # Reviewed games are development/training only, never fresh holdouts.
    for game in ([] if append else diagnostics['games']):
        f=game['first_costly_move']
        if f:add_root(chess.Board(f['fen']),'train',game['game_id'],'confirmed_failure',chess.Move.from_uci(f['played']))
    starts=json.loads((starts_path or ROOT/'benchmarks/openings-balanced-v19-data.json').read_text())
    for i,start in enumerate(starts):
        index=game_offset+i
        split='train' if index%6<4 else 'val' if index%6==4 else 'test';gid=7200000+index
        board=chess.Board(start.get('fen',chess.STARTING_FEN))
        for uci in start.get('moves',[]):board.push_uci(uci)
        for ply in range(64):
            if board.is_game_over():break
            if ply%8==0:add_root(board,split,gid,'ordinary_teacher_game')
            info=exact(board,8000);board.push(info['pv'][0])
        # Disallow subsequent source games from reusing roots/endpoints.
        seen.update(owners)
        if (i+1)%10==0:print('Broad source games',i+1,{s:len(pairs[s]) for s in pairs},flush=True)
    NEW.mkdir(exist_ok=append);(NEW/'pairs').mkdir(exist_ok=append)
    for split in rows:
        for path,items in [(NEW/f'{split}.jsonl',list(rows[split].values())),(NEW/'pairs'/f'{split}.jsonl',pairs[split])]:
            path.write_text(''.join(json.dumps(r)+'\n' for r in items))
    counts={s:{'rows':len(rows[s]),'pairs':len(pairs[s]),'old_correct_pairs':sum(r['old_gap_cp']>0 for r in pairs[s]),
        'confirmed_failure_pairs':sum(r['source']=='confirmed_failure' for r in pairs[s])} for s in rows}
    manifest={'counts':counts,'source_games':game_offset+len(starts),'seed':190001,'stockfish_sha256':digest(SF),
        'policy':'120 independent teacher source games,8 roots/game, whole-game4/1/1split; 64knode exact endpoint labels after tactical settling,200ms root confirmation. All prior data aliases excluded. Ordinary quiet score labels and old-correct rankings retained. Reviewed losses training only.'}
    write(NEW/'manifest.json',manifest);write(NEW/'pairs/manifest.json',manifest)
    return manifest


def prepare():
    DATA.mkdir();(DATA/'pairs').mkdir();counts={}
    for split in ('train','val','test'):
        sources=[BASE,NEW] if split=='train' else [NEW]
        for rel,fields in [(f'{split}.jsonl',('fen',)),(f'pairs/{split}.jsonl',('good_fen','bad_fen'))]:
            records={}
            for source in sources:
                for row in readrows(source/rel):records[tuple(key(row[f]) for f in fields)]=row
            (DATA/rel).write_text(''.join(json.dumps(r)+'\n' for r in records.values()))
        counts[split]={'rows':len(readrows(DATA/f'{split}.jsonl')),'pairs':len(readrows(DATA/'pairs'/f'{split}.jsonl'))}
    manifest={'counts':counts,'policy':'Retain all previous training examples and broaden with ordinary/old-correct/confirmed cases. Fresh new whole-game val/test only; all prior heldouts excluded from fitting.',
        'sources':{str(p):digest(p/'manifest.json') for p in (BASE,NEW)}}
    write(DATA/'manifest.json',manifest);write(DATA/'pairs/manifest.json',manifest)


def choose_weight(records,checkpoint):
    metrics={str(w):pair_metrics(records,NeuralEvaluator(checkpoint,w,True,incremental=True)) for w in (.05,.1,.25)}
    chosen=max((.05,.1,.25),key=lambda w:(metrics[str(w)]['accuracy'],-metrics[str(w)]['previously_correct_regressions'],-w))
    return chosen,metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-collection',action='store_true');args=parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    if args.resume_collection:
        if CP.exists() or DATA.exists() or json.loads((ART/'status.json').read_text())['phase']!='failed':raise ValueError('Resume only undersized collection before fitting')
        write(ART/'initial-status.json',json.loads((ART/'status.json').read_text()))
    elif (ART/'status.json').exists():raise FileExistsError('Preserve prior experiment')
    protocol={'diagnostics':'All24 v18 losses; first100cp/mate mistake confirmed; exactdepth3 and250ms comparisons across HCE/old/v18.',
        'data':'120new book starts,80train/20val/20testgames,8 roots/game. Ordinary teacher score/rank pairs, old-correct anchors, confirmed losses. Exclude all prior dataset aliases; whole-game isolation.',
        'training':'One v8-initialized model, samearchitecture,25% quiet rounded hybrid objective,lr.00005,rank2,score.1,anchor8,protected6,hard1,margin5,20epochs/patience5. Prior training retained.',
        'selection':'Validation only selects epoch and one of5/10/25% correction; accuracy then fewer heuristic regressions then smaller weight. Freeze hashes+weight before one test scoring and games. No test tuning.',
        'games':'Chosen model20freshpaired games vsHCE0 then20vsoldv8nonincremental25,250ms/depth8/max1000plies. Noapp promotion. NoStockfish opponent matches.'}
    write(ART/'protocol.json',protocol)
    def status(phase,**extra):write(ART/'status.json',{'phase':phase,**extra});print(phase,flush=True)
    def run(phase,args):status(phase,command=args);subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    try:
        with exclusive_cpu('v19 balanced training, diagnostics and isolated matches'):
            with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
                sf.configure({'Threads':1,'Hash':32})
                if args.resume_collection:
                    diagnostics=json.loads((ART/'diagnostics.json').read_text());manifest=json.loads((NEW/'manifest.json').read_text())
                else:
                    status('diagnose-v18-losses');diagnostics=diagnose(sf)
                    run('freeze-broad-source-starts',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-balanced-v19-data.json','--pairs','120','--seed','190001','--exclude-data',str(BASE)])
                    status('collect-broad-settled-supervision');manifest=collect(sf,diagnostics)
                if min(manifest['counts'][s]['pairs'] for s in ('val','test'))<30:
                    extra=ROOT/'benchmarks/openings-balanced-v19-data-extension.json'
                    run('freeze-extra-source-starts',['-m','benchmarks.generate_strength_openings','--output',str(extra),'--pairs','120','--seed','190002','--exclude-data',str(NEW)])
                    status('supplement-broad-supervision');manifest=collect(sf,diagnostics,extra,120,True)
            if min(manifest['counts'][s]['pairs'] for s in ('val','test'))<30:raise ValueError('Insufficient fresh cases; preserve before training')
            status('prepare-retained-training');prepare()
            run('train-rounded-25-percent',['-m','ml.train','--data',str(DATA),'--pairs',str(DATA/'pairs'),'--checkpoint',str(CP),'--metrics',str(ART/'training.json'),
                '--features','relationships','--target','residual','--color-consistent','--correction-limit-cp','250','--correction-weight','.25','--quiet-only',
                '--initial-checkpoint',str(OLD),'--learning-rate','.00005','--rank-weight','2','--rank-margin-cp','5','--hard-pair-weight','1',
                '--protected-pair-weight','6','--score-weight','.1','--anchor-weight','8','--selection','ranking','--epochs','20','--patience','5'])
            status('development-weight-screen')
            weight,dev=choose_weight(readrows(NEW/'pairs/val.jsonl'),CP)
            selection={'weight':weight,'quiet_only':True,'candidate_sha256':digest(CP),'validation_metrics':dev,'old_validation':pair_metrics(readrows(NEW/'pairs/val.jsonl'),NeuralEvaluator(OLD,.25,True)),
                'policy':protocol['selection'],'frozen_before_test_and_games':True}
            write(ART/'frozen-selection.json',selection)
            records=readrows(NEW/'pairs/test.jsonl')
            write(ART/'test.json',{'candidate':pair_metrics(records,NeuralEvaluator(CP,weight,True)),'old':pair_metrics(records,NeuralEvaluator(OLD,.25,True)),
                'heuristic':pair_metrics(records,NeuralEvaluator(OLD,0,True)),'selected_weight':weight,'policy':'One frozen configuration; test never chooses weight or epoch.'})
            run('freeze-fresh-game-starts',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-balanced-v19-games.json','--pairs','10','--seed','192500','--exclude-data',str(DATA)])
            matches={}
            for name,extra in [('vs-heuristic',['--opponent-nn-weight','0','--opponent-incremental']),('vs-old-NN',['--opponent-nn-weight','.25'])]:
                output=ART/name
                run(name,['-m','ml.compare','--checkpoint',str(CP),'--nn-weight',str(weight),'--quiet-only','--incremental','--opponent-checkpoint',str(OLD),
                    '--opponent-quiet-only','--openings','benchmarks/openings-balanced-v19-games.json','--pairs','10','--time-ms','250','--max-plies','1000','--output',str(output),*extra])
                matches[name]=json.loads((output/'report.json').read_text())['summary']
            status('completed',matches=matches,selected_weight=weight,app_promoted=False)
    except Exception as exc:status('failed',error=str(exc));raise


if __name__=='__main__':main()
