"""Capacity-aware search-leaf training, exact inference speedup, frozen games."""
from functools import lru_cache
import json
from pathlib import Path
import random
import subprocess
import sys
import chess
import chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize
from benchmarks.nn_baseline_v20.evaluator import NeuralEvaluator as FrozenEvaluator
from engine.evaluation import evaluate
from ml.dataset import correction_factor
from ml.evaluator import NeuralEvaluator
from ml.evaluate_blends import pair_metrics
from ml.generate_search_data import key
from ml.run_balanced_v19 import readrows,write,digest

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/leaf-speed-v20'
NEW=ROOT/'ml/data/leaf-speed-v20-new'
DATA=ROOT/'ml/data/leaf-speed-v20-2026'
BASE=ROOT/'ml/data/balanced-v19-2026'
OLD=ROOT/'ml/artifacts/quiet-ranking-v8.pt'
CP=ART/'trained.pt'
SF=ROOT/'tools/stockfish-sf19/stockfish/stockfish-macos-universal'
WEIGHT=.05
canonical=lru_cache(maxsize=200000)(key)


@lru_cache(maxsize=100000)
def baseline(fen):return evaluate(chess.Board(fen))


def ranking_range(good_fen,bad_fen,sign,weight=WEIGHT):
    bounds=[]
    for fen in (good_fen,bad_fen):
        board=chess.Board(fen);base=baseline(fen);limit=250*correction_factor(board,weight,True)
        bounds.append((round(base-limit),round(base+limit)))
    a,b=bounds
    return (a[0]-b[1],a[1]-b[0]) if sign==1 else (b[0]-a[1],b[1]-a[0])


def collect():
    rng=random.Random(200020);prior=set()
    for path in (ROOT/'ml/data').glob('**/*.jsonl'):
        for row in readrows(path):
            prior.update(canonical(row[f]) for f in ('fen','root_fen','good_fen','bad_fen') if f in row)
    groups={};seen_roots=set()
    for record in readrows(ART/'leaves.jsonl'):
        k=canonical(record['fen']);gid=record['game_id']
        if k in prior or (gid,k) in seen_roots:continue
        seen_roots.add((gid,k));groups.setdefault(gid,[]).append(record)
    for records in groups.values():rng.shuffle(records)
    rows={s:{} for s in ('train','val','test')};pairs={s:[] for s in rows};owners={};pair_seen=set();stats={'roots':0,'candidate_endpoints':0,'outside_correction_range':0,'teacher_gap_too_small':0}
    cache={}
    with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32})
        def exact(board,nodes=64000):
            k=(board.fen(),nodes)
            if k in cache:return cache[k]
            last=None
            with sf.analysis(board,chess.engine.Limit(nodes=nodes),game=object()) as stream:
                for info in stream:
                    if info.get('pv') and 'score' in info and not info.get('lowerbound') and not info.get('upperbound'):last=dict(info)
            if last is None:raise ValueError('No exact scored teacher PV')
            cache[k]=last;return last
        def settle(board):
            board=board.copy(stack=False);line=[]
            for _ in range(17):
                if board.is_game_over():return None
                move=exact(board,12000)['pv'][0]
                if not (board.is_check() or board.is_capture(move) or move.promotion):return board,line
                line.append(move.uci());board.push(move)
            return None
        def add(record):
            board=chess.Board(record['fen']);gid=record['game_id'];split=record['split'];rootkey=canonical(board.fen())
            if board.is_game_over() or owners.get(rootkey,gid)!=gid:return
            stats['roots']+=1
            infos=sf.analyse(board,chess.engine.Limit(nodes=24000),multipv=3,game=object())
            if not infos or not infos[0].get('pv'):return
            good_move=infos[0]['pv'][0];a=board.copy();a.push(good_move);good=settle(a)
            if good is None:return
            candidates=[info['pv'][0] for info in infos[1:] if info.get('pv')]
            candidates+=rng.sample(list(board.legal_moves),min(2,board.legal_moves.count()))
            for move in dict.fromkeys(candidates):
                if move==good_move:continue
                b=board.copy();b.push(move);bad=settle(b)
                if bad is None:continue
                good_fen,bad_fen=good[0].fen(),bad[0].fen();sign=1 if board.turn else -1
                factor=sum(correction_factor(leaf,WEIGHT,True) for leaf in (good[0],bad[0]))
                if factor==0:continue
                hgap=sign*(baseline(good_fen)-baseline(bad_fen));low,high=ranking_range(good_fen,bad_fen,sign)
                # Sample consequential near-ties the actual bounded, rounded
                # configuration can improve, and vulnerable correct rankings.
                if high<5 or hgap>0 and low>5:
                    stats['outside_correction_range']+=1;continue
                keys=canonical(good_fen),canonical(bad_fen)
                if keys[0]==keys[1] or set(keys)&prior or any(owners.get(k,gid)!=gid for k in keys):continue
                if keys in pair_seen:continue
                ai,bi=exact(good[0]),exact(bad[0]);acp,bcp=ai['score'].white().score(),bi['score'].white().score()
                if acp is None or bcp is None or max(abs(acp),abs(bcp))>=1500:continue
                gap=sign*(acp-bcp)
                if gap<50:
                    stats['teacher_gap_too_small']+=1;continue
                pair_seen.add(keys);owners[rootkey]=gid;stats['candidate_endpoints']+=2
                pairs[split].append({'root_fen':board.fen(),'good_fen':good_fen,'bad_fen':bad_fen,'sign':sign,
                    'game_id':gid,'source':'capacity_aware_quiet_search_leaf','cp_loss':gap,'heuristic_gap_cp':hgap,
                    'minimum_runtime_gap_cp':low,'maximum_runtime_gap_cp':high,
                    'good_move':good_move.uci(),'bad_move':move.uci(),'good_tactical_line':good[1],'bad_tactical_line':bad[1],
                    'label_nodes':64000,'training_weight':12 if split=='train' and hgap<=0 else 3 if split=='train' else 1})
                for leaf,cp in ((good[0],acp),(bad[0],bcp)):
                    k=canonical(leaf.fen());owners[k]=gid
                    rows[split][k]={'fen':leaf.fen(),'score_cp':cp,'game_id':gid,'ply':leaf.ply(),'source':'capacity_aware_quiet_search_leaf','root_fen':board.fen()}
            # A fixed quarter of sampled roots supplies ordinary quiet labels.
            if stats['roots']%4==0 and owners.get(rootkey,gid)==gid:
                cp=exact(board)['score'].white().score()
                if cp is not None and abs(cp)<1500:
                    rows[split][rootkey]={'fen':board.fen(),'score_cp':cp,'game_id':gid,'ply':board.ply(),'source':'ordinary_quiet_search_leaf'};owners[rootkey]=gid
        for pass_index in range(2):
            for gid,records in sorted(groups.items()):
                for record in records[pass_index*80:(pass_index+1)*80]:add(record)
                print('Leaf group',gid,'pass',pass_index+1,{s:len(pairs[s]) for s in pairs},flush=True)
            if min(len(pairs[s]) for s in ('val','test'))>=30:break
            write(ART/'coverage-extension.json',{'reason':'Fixed next80 roots/group permitted before fitting when a holdout has fewer than30pairs','candidate_fitted':False})
    NEW.mkdir();(NEW/'pairs').mkdir()
    counts={}
    for split in rows:
        for path,records in [(NEW/f'{split}.jsonl',list(rows[split].values())),(NEW/'pairs'/f'{split}.jsonl',pairs[split])]:
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        counts[split]={'rows':len(rows[split]),'pairs':len(pairs[split]),'correctable_heuristic_wrong_pairs':sum(r['heuristic_gap_cp']<=0 for r in pairs[split]),
            'opening_groups':len({r['game_id'] for r in pairs[split]})}
    manifest={'counts':counts,'statistics':stats,'seed':200020,'stockfish_sha256':digest(SF),
        'policy':'Real quiet static search calls from v19 games. Same opening pair/all colors/opponents owns one split (6train/2val/2testopeninggroups). Up to80 roots/group, fixed next80 only if undersized before fitting. Exact64knode labels after12knode teacher tactical settling,>=50cp settled gap. Sample rounded5% reachable near-ties and vulnerable correct rankings. Canonical prior aliases excluded; targeted holdout is not general accuracy.'}
    write(NEW/'manifest.json',manifest);write(NEW/'pairs/manifest.json',manifest)
    if min(counts[s]['pairs'] for s in ('val','test'))<30:raise ValueError('Insufficient targeted holdout; preserve data before fitting')


def prepare():
    DATA.mkdir();(DATA/'pairs').mkdir();removed=0;counts={}
    for split in ('train','val','test'):
        sources=[BASE,NEW] if split=='train' else [NEW]
        scores={};rankings={}
        for source in sources:
            for row in readrows(source/f'{split}.jsonl'):scores[canonical(row['fen'])]=row
            for row in readrows(source/'pairs'/f'{split}.jsonl'):
                if split=='train' and row['heuristic_gap_cp']<=0 and ranking_range(row['good_fen'],row['bad_fen'],row['sign'])[1]<5:
                    removed+=1;continue
                rankings[(canonical(row['good_fen']),canonical(row['bad_fen']))]=row
        (DATA/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores.values()))
        (DATA/'pairs'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rankings.values()))
        counts[split]={'rows':len(scores),'pairs':len(rankings)}
    manifest={'counts':counts,'unattainable_previous_training_rankings_removed':removed,
        'policy':'Prior score labels and attainable/correct prior training rankings retained. Inert hard training margins excluded using exact rounded5% bounds; heldout sampling frozen before fitting. Fresh new capacity-targeted val/test only.',
        'sources':{str(p):digest(p/'manifest.json') for p in (BASE,NEW)}}
    write(DATA/'manifest.json',manifest);write(DATA/'pairs/manifest.json',manifest)


def main():
    if (ART/'status.json').exists():raise FileExistsError('Preserve experiment')
    protocol={'speed':'Opt-in shared attacks, original-coordinate feature canonicalization and reusable NumPy buffers. Frozen b962422 inference/unchanged engine reference. Same5% weights; require integer/search-tree parity, alternating-order profiles.',
        'data':'Capacity-aware real quiet search leaves, paired-opening groups isolated.>=50cp exact64knode settled teacher ranking; actual rounded5% range determines useful examples. Retain ordinary/correct positions.80roots/group thenfixed80more if needed BEFORE fitting.',
        'training':'One v8-initialized unchanged architecture at5%quiet roundedhybrid. lr.0001 rank4 margin5 hard3 protected3 score.02 anchor.5 max25epochs patience6. Prior attainable/correct rankings retained.',
        'selection':'Train checkpoint selected on fresh targeted validation. Require strict validation accuracy gain and no extra heuristic-correct regressions; testnonregression veto only. If not qualified, predeclared speed-only v8 at5% is tested instead. No weight tuning; test never changes learned weights.',
        'games':'Freeze checkpoint/hash/5%quietfast mode before games.20freshpaired vsfrozenoptimizedHCE0,20vsfrozenoldv8nonincremental25 at250ms/depth8/1000plies. If vsHCE>=60% ANDvsold>50% andall40complete/errorfree, extend HCE to100total on40newpairs; noNNapp promotion orStockfishgames.'}
    write(ART/'protocol.json',protocol)
    def status(phase,**extra):write(ART/'status.json',{'phase':phase,**extra});print(phase,flush=True)
    def run(phase,args):status(phase,command=args);subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    try:
        with exclusive_cpu('v20 capacity-aware labels training and isolated games'):
            status('label-capacity-aware-leaves');collect();status('retain-useful-training');prepare()
            run('train-actual-5-percent',['-m','ml.train','--data',str(DATA),'--pairs',str(DATA/'pairs'),'--checkpoint',str(CP),'--metrics',str(ART/'training.json'),
                '--features','relationships','--target','residual','--color-consistent','--correction-limit-cp','250','--correction-weight','.05','--quiet-only',
                '--initial-checkpoint',str(OLD),'--learning-rate','.0001','--rank-weight','4','--rank-margin-cp','5','--hard-pair-weight','3',
                '--protected-pair-weight','3','--score-weight','.02','--anchor-weight','.5','--selection','ranking','--epochs','25','--patience','6'])
            status('qualify-on-frozen-targeted-positions')
            cases={s:readrows(NEW/'pairs'/f'{s}.jsonl') for s in ('val','test')}
            metrics={name:{s:pair_metrics(rows,model) for s,rows in cases.items()} for name,model in {
                'old5':FrozenEvaluator(OLD,.05,True,incremental=True),'trained5':NeuralEvaluator(CP,.05,True,incremental=True,fast_features=True),
                'old25':FrozenEvaluator(OLD,.25,True),'heuristic':FrozenEvaluator(OLD,0,True,incremental=True)}.items()}
            a,b=metrics['old5'],metrics['trained5']
            qualified=(b['val']['accuracy']>a['val']['accuracy'] and b['val']['previously_correct_regressions']<=a['val']['previously_correct_regressions']
                and b['test']['accuracy']>=a['test']['accuracy'] and b['test']['previously_correct_regressions']<=a['test']['previously_correct_regressions'])
            selected=CP if qualified else OLD
            write(ART/'frozen-selection.json',{'trained_qualified':qualified,'checkpoint':str(selected),'checkpoint_sha256':digest(selected),
                'nn_weight':WEIGHT,'quiet_only':True,'incremental':True,'fast_features':True,'metrics':metrics,'policy':protocol['selection']})
            run('freeze-pilot-openings',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-leaf-speed-v20-pilot.json','--pairs','10','--seed','202500','--exclude-data',str(DATA)])
            common=['-m','ml.compare','--checkpoint',str(selected),'--nn-weight','.05','--quiet-only','--incremental','--fast-features',
                '--opponent-checkpoint',str(OLD),'--opponent-quiet-only','--opponent-frozen-inference','--openings','benchmarks/openings-leaf-speed-v20-pilot.json','--pairs','10','--time-ms','250','--max-plies','1000']
            matches={}
            for name,extra in [('vs-heuristic',['--opponent-nn-weight','0','--opponent-incremental']),('vs-old-NN',['--opponent-nn-weight','.25'])]:
                output=ART/name;run(name,[*common,*extra,'--output',str(output)]);matches[name]=json.loads((output/'report.json').read_text())['summary']
            extend=(all(s['completed']==20 and not s['errors'] for s in matches.values()) and matches['vs-heuristic']['score_fraction_completed']>=.6 and matches['vs-old-NN']['score_fraction_completed']>.5)
            if extend:
                run('freeze-extension-openings',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-leaf-speed-v20-extension.json','--pairs','40','--seed','202501','--exclude-data',str(DATA)])
                output=ART/'vs-heuristic-extension'
                run('extend-to-100',['-m','ml.compare','--checkpoint',str(selected),'--nn-weight','.05','--quiet-only','--incremental','--fast-features','--opponent-checkpoint',str(OLD),
                    '--opponent-nn-weight','0','--opponent-quiet-only','--opponent-incremental','--opponent-frozen-inference','--openings','benchmarks/openings-leaf-speed-v20-extension.json','--pairs','40','--time-ms','250','--max-plies','1000','--output',str(output)])
                games=json.loads((ART/'vs-heuristic/report.json').read_text())['games'];extra=json.loads((output/'report.json').read_text())['games']
                for game in extra:game['pair']+=10
                matches['vs-heuristic-100']=summarize(games+extra)
            status('completed',matches=matches,extended_to_100=extend,trained_qualified=qualified,checkpoint=str(selected),app_promoted=False)
    except Exception as exc:status('failed',error=str(exc));raise


if __name__=='__main__':main()
