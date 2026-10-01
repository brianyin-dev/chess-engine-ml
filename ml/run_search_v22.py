"""Actual search leaves, smooth protected training, move regret then strength."""
import hashlib,json,random,subprocess,sys
from collections import defaultdict
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize
from engine.search import search
from ml.evaluator import NeuralEvaluator
from ml.profile_leaf_speed_v20 import Leaves
from ml.run_balanced_v19 import write,readrows,digest
from ml import run_leaf_speed_v20 as labels
from ml.train_search_v22 import train
ROOT=labels.ROOT;ART=ROOT/'ml/artifacts/search-smooth-v22';NEW=ROOT/'ml/data/search-smooth-v22-new';DATA=ROOT/'ml/data/search-smooth-v22-2026';OLD=labels.OLD;CP=ART/'trained.pt'


def identity(initial,opening):
    b=chess.Board(initial)
    for u in opening[:4]:b.push_uci(u)
    gid=int(hashlib.sha256(labels.canonical(b.fen()).encode()).hexdigest()[:12],16)
    return gid,('train' if gid%10<6 else 'val' if gid%10<8 else 'test')


def sample():
    rng=random.Random(220022);roots=[];families=defaultdict(list)
    # Whole opening family owns every variant/color/source, including old matches.
    for path in sorted((ROOT/'ml/artifacts').glob('*/vs*/report.json')):
        for gi,g in enumerate(json.loads(path.read_text()).get('games',[])):
            gid,split=identity(g['initial_fen'],g['opening_moves'])
            for fraction in (.25,.5,.75):
                b=chess.Board(g['initial_fen'])
                for u in g['opening_moves']:b.push_uci(u)
                for item in g['moves'][:int(len(g['moves'])*fraction)]:b.push_uci(item['uci'])
                if b.is_game_over():continue
                families[gid].append({'initial_fen':g['initial_fen'],'history':[m.uci() for m in b.move_stack],'fen':b.fen(),'game_id':gid,'split':split,'origin':'heuristic_NN_match','source':str(path.relative_to(ROOT)),'source_game':gi})
    # Broaden opening-family exposure with prior source games, but sample actual
    # evaluations from OUR search, never reuse teacher labels as new examples.
    for path in [ROOT/'ml/artifacts/generalization-v21/source-games.json',ROOT/'ml/artifacts/generalization-v21/coverage-extension/source-games.json']:
        counts=defaultdict(int)
        for g in json.loads(path.read_text()):
            gid,split=identity(chess.STARTING_FEN,g['opening'])
            if counts[gid]>=2:continue
            counts[gid]+=1;b=chess.Board()
            history=g['opening']+g['moves'][:len(g['moves'])//2]
            for u in history:b.push_uci(u)
            if b.is_game_over():continue
            families[gid].append({'initial_fen':chess.STARTING_FEN,'history':history,'fen':b.fen(),'game_id':gid,'split':split,'origin':'book_source_our_search','source':str(path.relative_to(ROOT)),'source_game':g['game_id']})
    models={'heuristic':NeuralEvaluator(OLD,0,True,incremental=True,fast_features=True),'old25':NeuralEvaluator(OLD,.25,True,incremental=True,fast_features=True)}
    all_leaves=[];selected_roots=[]
    for gid,cases in sorted(families.items()):
        rng.shuffle(cases);records=[];seen=set()
        # Up to6 roots/family; all actual match origins get priority.
        cases=sorted(cases,key=lambda r:r['origin']!='heuristic_NN_match')[:6]
        for record in cases:
            b=chess.Board(record['initial_fen'])
            for u in record['history']:b.push_uci(u)
            selected_roots.append(record);ri=len(selected_roots)-1
            for name,model in models.items():
                sampler=Leaves(model,rng);search(b.copy(stack=True),depth=3,node_limit=2000,eval_fn=sampler)
                for fen in sampler.fens:
                    k=labels.canonical(fen)
                    if k in seen:continue
                    seen.add(k);records.append({'fen':fen,'game_id':gid,'split':record['split'],'root_index':ri,'sampler':name,'origin':record['origin']})
        rng.shuffle(records);all_leaves.extend(records[:24])
        print('Sample family',gid,'rootcount',len(cases),'leaves',min(len(records),24),flush=True)
    write(ART/'roots.json',selected_roots)
    (ART/'leaves.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_leaves))
    write(ART/'sampling.json',{'roots':len(selected_roots),'leaves':len(all_leaves),'families':len(families),'quiet_leaf_policy':'Actual quiet evaluation calls, reservoir sampling40 per search then canonical unique24/family. Both heuristic andold25, depth3/node2000 collector only, never used for strength/speed claims. Same family owns all colors/evaluator variants.','origin_counts':{n:sum(r['origin']==n for r in all_leaves) for n in ('heuristic_NN_match','book_source_our_search')}})


def prepare():
    DATA.mkdir();(DATA/'pairs').mkdir();counts={};sets={s:set() for s in ('train','val','test')}
    for s in sets:
        scores=readrows(NEW/f'{s}.jsonl');pairs=readrows(NEW/'pairs'/f'{s}.jsonl')
        if s=='train':
            # Retain prior training; heldouts never enter adaptation training.
            base=ROOT/'ml/data/generalization-v21-2026'
            scores=readrows(base/'train.jsonl')+scores;pairs=readrows(base/'pairs/train.jsonl')+pairs
        for folder,rows in [(DATA,scores),(DATA/'pairs',pairs)]:
            (folder/f'{s}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for r in scores+pairs:sets[s].update(labels.canonical(r[f]) for f in ('fen','root_fen','good_fen','bad_fen') if f in r)
        counts[s]={'rows':len(scores),'pairs':len(pairs),'new_families':len({r['game_id'] for r in readrows(NEW/'pairs'/f'{s}.jsonl')})}
    overlap={f'{a}-{b}':len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(overlap.values())
    write(DATA/'manifest.json',{'counts':counts,'cross_split_aliases':overlap,'policy':'Actual quiet leaves from our search. New source families isolated by first4ply canonical position. Legacy prior training retained; prior canonical labels excluded fromnewdata. Legacy opening-family provenance is incomplete, so no claim of globally unseen openings for pretrained/retained legacy examples.'})
    write(DATA/'pairs/manifest.json',json.loads((DATA/'manifest.json').read_text()))


def exact(sf,b,root_moves=None,nodes=64000):
    last=None
    with sf.analysis(b,chess.engine.Limit(nodes=nodes),root_moves=root_moves,game=object()) as stream:
        for i in stream:
            if i.get('pv') and 'score' in i and not i.get('lowerbound') and not i.get('upperbound'):last=dict(i)
    if last is None:raise ValueError('Missing exact teacherPV')
    return last


def freeze_screen():
    # Test roots are frozen before fitting, not chosen by mistakes or candidate.
    roots=json.loads((ART/'roots.json').read_text());rng=random.Random(220023);rng.shuffle(roots)
    prior=set()
    for path in (ROOT/'ml/data').glob('**/*.jsonl'):
        for r in readrows(path):prior.update(labels.canonical(r[f]) for f in ('fen','root_fen','good_fen','bad_fen') if f in r)
    cases=[];seen=set();perfamily=defaultdict(int)
    for r in roots:
        k=labels.canonical(r['fen'])
        if r['split']!='test' or k in prior or k in seen or perfamily[r['game_id']]>=2:continue
        seen.add(k);perfamily[r['game_id']]+=1;cases.append(r)
        if len(cases)==32:break
    if len(cases)<16:raise ValueError('Insufficient novel test roots')
    write(ART/'frozen-move-screen.json',cases)


def screen():
    cases=json.loads((ART/'frozen-move-screen.json').read_text())
    models={'heuristic':NeuralEvaluator(OLD,0,True,incremental=True,fast_features=True),'old25':NeuralEvaluator(OLD,.25,True,incremental=True,fast_features=True),'trained25':NeuralEvaluator(CP,.25,True,incremental=True,fast_features=True)}
    result=[]
    with chess.engine.SimpleEngine.popen_uci(str(labels.SF)) as sf:
        sf.configure({'Threads':1,'Hash':32})
        for r in cases:
            b=chess.Board(r['initial_fen'])
            for u in r['history']:b.push_uci(u)
            if b.is_game_over():continue
            teacher=exact(sf,b);best=teacher['score'].pov(b.turn).score();probes={};cache={}
            for name,model in models.items():
                move=search(b.copy(stack=True),depth=3,eval_fn=model)
                assert move.depth==3
                u=move.move.uci()
                if u not in cache:
                    info=teacher if move.move==teacher['pv'][0] else exact(sf,b,[move.move])
                    cache[u]={'cp':info['score'].pov(b.turn).score(),'mate':info['score'].pov(b.turn).mate(),'depth':info.get('depth')}
                sc=cache[u];regret=max(0,best-sc['cp']) if best is not None and sc['cp'] is not None else None
                probes[name]={'move':u,'depth':move.depth,'nodes':move.nodes,'qnodes':move.qnodes,'teacher':sc,'regret_cp':regret,'allows_mate':sc['mate'] is not None and sc['mate']<0 and teacher['score'].pov(b.turn).mate() is None}
            result.append({'root':r,'teacher_best':teacher['pv'][0].uci(),'probes':probes});print('Move screen',len(result),{n:p['regret_cp'] for n,p in probes.items()},flush=True)
    comparable=[r for r in result if all(p['regret_cp'] is not None for p in r['probes'].values())]
    summary={n:{'mean_cp_regret':sum(r['probes'][n]['regret_cp'] for r in comparable)/len(comparable),'bad_100cp_moves':sum(r['probes'][n]['regret_cp']>=100 for r in comparable),'allows_mate':sum(r['probes'][n]['allows_mate'] for r in result)} for n in models}
    qualified=(len(comparable)>=16 and summary['trained25']['mean_cp_regret']<summary['heuristic']['mean_cp_regret'] and summary['trained25']['mean_cp_regret']<=summary['old25']['mean_cp_regret'] and summary['trained25']['bad_100cp_moves']<=summary['heuristic']['bad_100cp_moves'] and summary['trained25']['allows_mate']<=summary['heuristic']['allows_mate'])
    write(ART/'move-screen.json',{'cases':result,'comparable':len(comparable),'summary':summary,'qualified_for_games':qualified,'policy':'Fixed32 novel test-family roots,max2/family; exact depth3 allmodels/no clock cap. Same-root64knode last-exact teacher score, forcedmove64knodes; comparable nonmate scores only, mate events separately. This gate makes screen development; only subsequent freshgames can confirm strength.'})
    return qualified


def main():
    if ART.exists():raise FileExistsError('Preserve v22')
    ART.mkdir()
    def phase(name,**extra):write(ART/'status.json',{'phase':name,**extra});print(name,flush=True)
    def run(args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    protocol={'data':'Quiet runtime calls from historicalheuristicNNgames plus oursearch from560booksourcegames; family-isolated currentadaptation.24uniqueleaves/family; bothHCEandold25 searches. Teacher settled64knode endpoint and ordinary quietlabels; actualrounded25% reachable5cpmargins. Allpriorlabels excluded; retain oldtrain only.','training':'One unchanged64x32 model at25% quiet,v20fastfeatures. Smooth teacher expected-score proxy, logisticgap targets, protection foroldNNcorrectrankings8, anchor2; lr.00005,30epochs/patience8,seed42. Bestupdatedepoch byvalcorrect minus2oldregressions thenaccuracy thenproxyMSE. No silentepochzero fallback.','screen':'Freeze novel test-familyroots before fitting;32max/2perfamily. Evaluate allmodels exactdepth3 thenStockfish64knode same-rootregret. Requirelowerregretvsheuristic andnotworsevsoldNN, noextrablunders/mates. Screenisdevelopmentgate, notfinalholdoutclaim.','games':'If screenpassed:20freshpairedgames250ms/depth8 vsunchangedheuristic.>=60%/complete/errorfree extends80freshgames to100total; reportextensionseparately; oldNNsecondary20 onlyifpilotpassed. Noapppromotion.'}
    write(ART/'protocol.json',protocol)
    write(ART/'fixed-source-hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'engine/search.py',ROOT/'engine/evaluation.py',ROOT/'ml/model.py',ROOT/'ml/evaluator.py',ROOT/'ml/incremental.py']})
    try:
        with exclusive_cpu('v22 actual search leaves smooth training strength'):
            phase('sample-actual-runtime-leaves');sample()
            labels.ART=ART;labels.NEW=NEW;labels.WEIGHT=.25
            original=labels.ranking_range;labels.ranking_range=lambda a,b,s,weight=.25:original(a,b,s,weight)
            phase('label-settled-runtime-decisions');labels.collect()
            manifest=json.loads((NEW/'manifest.json').read_text());manifest['policy']=protocol['data'];write(NEW/'manifest.json',manifest);write(NEW/'pairs/manifest.json',manifest)
            phase('prepare-and-audit');prepare();freeze_screen()
            phase('smooth-protected-training');train(DATA,CP,ART/'training.json',OLD)
            phase('equal-depth-move-regret');qualified=screen()
            if not qualified:phase('completed',qualified_for_games=False,matches={},app_promoted=False);return
            selection={'checkpoint_sha256':digest(CP),'weight':.25,'quiet_only':True,'incremental':True,'fast_features':True};write(ART/'frozen-selection.json',selection)
            opening='benchmarks/openings-search-smooth-v22-pilot.json';run(['-m','benchmarks.generate_strength_openings','--output',opening,'--pairs','10','--seed','222500','--exclude-data',str(DATA)])
            common=['-m','ml.compare','--checkpoint',str(CP),'--nn-weight','.25','--quiet-only','--incremental','--fast-features','--opponent-checkpoint',str(OLD),'--opponent-quiet-only','--opponent-frozen-inference','--time-ms','250','--max-plies','1000']
            out=ART/'vs-heuristic';phase('heuristic-pilot');run(common+['--opponent-nn-weight','0','--opponent-incremental','--openings',opening,'--pairs','10','--output',str(out)])
            report=json.loads((out/'report.json').read_text());matches={'pilot':report['summary']};extend=matches['pilot']['completed']==20 and not matches['pilot']['errors'] and matches['pilot']['score_fraction_completed']>=.6
            if extend:
                opening2='benchmarks/openings-search-smooth-v22-extension.json';run(['-m','benchmarks.generate_strength_openings','--output',opening2,'--pairs','40','--seed','222501','--exclude-data',str(DATA)])
                out2=ART/'vs-heuristic-extension';phase('100-game-confirmation');run(common+['--opponent-nn-weight','0','--opponent-incremental','--openings',opening2,'--pairs','40','--output',str(out2)])
                extra=json.loads((out2/'report.json').read_text());matches['extension']=extra['summary']
                for g in extra['games']:g['pair']+=10
                matches['combined100']=summarize(report['games']+extra['games'])
                out3=ART/'vs-old-NN';run(common+['--opponent-nn-weight','.25','--openings',opening,'--pairs','10','--output',str(out3)]);matches['oldNN']=json.loads((out3/'report.json').read_text())['summary']
            phase('completed',qualified_for_games=True,matches=matches,extended_to_100=extend,app_promoted=False)
    except Exception as e:phase('failed',error=str(e));raise
if __name__=='__main__':main()
