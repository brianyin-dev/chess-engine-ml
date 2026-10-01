"""Family-isolated supervision with fixed v20 inference and strength gates."""
import hashlib,json,random,subprocess,sys
from pathlib import Path
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize
from engine.opening_book import choose_book_move
from ml.evaluate_blends import pair_metrics
from ml.evaluator import NeuralEvaluator
from ml.dataset import correction_factor
from ml.run_balanced_v19 import write,readrows,digest
from ml import run_leaf_speed_v20 as previous
ROOT=previous.ROOT
ART=ROOT/'ml/artifacts/generalization-v21'
NEW=ROOT/'ml/data/generalization-v21-new'
DATA=ROOT/'ml/data/generalization-v21-2026'
OLD=previous.OLD
CP=ART/'trained.pt'

def diagnose():
    models={'old5':NeuralEvaluator(OLD,.05,True,incremental=True,fast_features=True),'v20trained5':NeuralEvaluator(previous.CP,.05,True,incremental=True,fast_features=True)}
    records=[]
    for split in ('val','test'):
        for r in readrows(previous.NEW/'pairs'/f'{split}.jsonl'):
            a,b=chess.Board(r['good_fen']),chess.Board(r['bad_fen'])
            gaps={name:r['sign']*(model(a)-model(b)) for name,model in models.items()}
            if gaps['v20trained5']>0:continue
            tags=[]
            if a.is_check() or b.is_check():tags.append('check_gate')
            if any(next(x.generate_legal_captures(),None) for x in (a,b)):tags.append('capture_gate')
            if any(len(x.piece_map())<=12 for x in (a,b)):tags.append('endgame')
            if any(len(x.pieces(chess.QUEEN,c)) for x in (a,b) for c in chess.COLORS):tags.append('queens_present')
            if any(len(x.pieces(chess.PAWN,c))>=5 for x in (a,b) for c in chess.COLORS):tags.append('pawn_structure')
            records.append({**r,'former_split':split,'runtime_gaps':gaps,'tags':tags,'regression_from_old':gaps['old5']>0})
    write(ART/'former-holdout-diagnostics.json',{'policy':'Descriptive overlapping board categories, not causal evidence. Former holdouts excluded from new training and evaluation. Already confirmed64knode settled labels reused; no new test-based selection.','mistakes':records,'counts':{tag:sum(tag in r['tags'] for r in records) for tag in sorted({t for r in records for t in r['tags']})},'regressions':sum(r['regression_from_old'] for r in records)})

def roots():
    rng=random.Random(210021);records=[];families={};seen=set();games=[]
    with chess.engine.SimpleEngine.popen_uci(str(previous.SF)) as sf:
        sf.configure({'Threads':1,'Hash':32})
        for gid in range(240):
            board=chess.Board();opening=[]
            for _ in range(12):
                choice=choose_book_move(board,ROOT/'books/gm2001.bin',rng=rng)
                if choice is None:break
                board.push(choice.move);opening.append(choice.move.uci())
            if len(opening)<4:continue
            # Transposing first-four-ply positions share a family and split.
            family=previous.canonical(chess.Board().fen())
            familyboard=chess.Board()
            for u in opening[:4]:familyboard.push_uci(u)
            family=previous.canonical(familyboard.fen())
            familyid=int(hashlib.sha256(family.encode()).hexdigest()[:12],16)
            bucket=familyid%10;split='train' if bucket<6 else 'val' if bucket<8 else 'test'
            families[family]={'id':familyid,'split':split}
            snapshots=[];moves=[]
            for ply in range(72):
                if board.is_game_over():break
                if ply>=6 and not board.is_check() and next(board.generate_legal_captures(),None) is None:
                    snapshots.append(board.fen())
                move=sf.play(board,chess.engine.Limit(nodes=2000),game=gid).move
                moves.append(move.uci());board.push(move)
            rng.shuffle(snapshots)
            for fen in snapshots[:8]:
                k=previous.canonical(fen)
                if k in seen:continue
                seen.add(k);records.append({'fen':fen,'game_id':familyid,'source_game':gid,'split':split})
            games.append({'game_id':gid,'family':family,'split':split,'opening':opening,'moves':moves})
            if gid%20==0:print('Source games',gid,'roots',len(records),'families',len(families),flush=True)
    (ART/'leaves.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(ART/'source-games.json',games);write(ART/'families.json',list(families.values()))
    if min(sum(r['split']==s for r in records) for s in ('train','val','test'))<80:raise ValueError('Insufficient split coverage')

def prepare():
    DATA.mkdir();(DATA/'pairs').mkdir();counts={};allaliases={s:set() for s in ('train','val','test')}
    for s in allaliases:
        for r in readrows(NEW/f'{s}.jsonl')+readrows(NEW/'pairs'/f'{s}.jsonl'):
            allaliases[s].update(previous.canonical(r[f]) for f in ('fen','root_fen','good_fen','bad_fen') if f in r)
    assert not any(allaliases[a]&allaliases[b] for a,b in [('train','val'),('train','test'),('val','test')])
    for s in allaliases:
        scores=readrows(NEW/f'{s}.jsonl');pairs=readrows(NEW/'pairs'/f'{s}.jsonl')
        if s=='train':
            scores=readrows(previous.DATA/'train.jsonl')+scores
            pairs=readrows(previous.DATA/'pairs/train.jsonl')+pairs
        for folder,rows in [(DATA,scores),(DATA/'pairs',pairs)]:
            (folder/f'{s}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        counts[s]={'rows':len(scores),'pairs':len(pairs),'new_families':len({r['game_id'] for r in readrows(NEW/'pairs'/f'{s}.jsonl')})}
    write(DATA/'manifest.json',{'counts':counts,'cross_split_aliases':0,'policy':'New opening families separated at first-four-ply canonical position; former heldouts excluded; all v20 training retained; fresh val/test only.'})
    write(DATA/'pairs/manifest.json',json.loads((DATA/'manifest.json').read_text()))

def main():
    resume='--resume-after-coverage' in sys.argv
    if ART.exists() and not resume:raise FileExistsError('Preserve v21')
    if resume:
        assert not CP.exists() and not DATA.exists()
        assert (ART/'coverage-extension/source-games.json').exists()
        assert min(len(readrows(NEW/'pairs'/f'{s}.jsonl')) for s in ('val','test'))>=30
    else:ART.mkdir()
    protocol={'inference':'Fixed v20 fast_features incremental quiet5%; unchanged architecture/search/heuristic.', 'data':'240 independent book-start Stockfish games,2000nodes/move72plies; up to8quiet roots/game; first4ply transposition families hash60/20/20 split.64knode settled ranking teacher and exact rounded5% capacity filter fromv20. Prior canonical labels excluded.','training':'One v8 initialized5% candidate; ranking4 margin5 hard3 protected3 lr.0001 anchor.1 score0 epochs30 patience8. Score loss disabled because target is useful bounded move decisions. Retain prior training rankings.','selection':'Validation ranking selects checkpoint; report fresh test untouched. Test is descriptive, not selector. Play learned candidate20games vsfrozenHCE regardless static qualification to prioritize strength; no unchanged checkpoint rerun. Extend to100total if pilot>=60%; extension separately must>50%, combined>=60% for success. Old NN20secondary only if heuristic pilot>=60%. No automatic app promotion.'}
    if not resume:write(ART/'protocol.json',protocol)
    def phase(name,**kw):write(ART/'status.json',{'phase':name,**kw});print(name,flush=True)
    def run(name,args):phase(name);subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    try:
        with exclusive_cpu('v21 independent families and strength'):
            if not resume:
                phase('diagnose-former-holdouts');diagnose();phase('generate-independent-games');roots()
                previous.ART=ART;previous.NEW=NEW
                phase('label-settled-useful-rankings')
                try:previous.collect()
                except ValueError as exc:
                    if 'Insufficient targeted holdout' not in str(exc):raise
                    run('pre-fit-family-coverage-extension',['-m','ml.extend_generalization_v21'])
            phase('audit-and-retain');prepare()
            run('train-move-decisions',['-m','ml.train','--data',str(DATA),'--pairs',str(DATA/'pairs'),'--checkpoint',str(CP),'--metrics',str(ART/'training.json'),'--features','relationships','--target','residual','--color-consistent','--correction-limit-cp','250','--correction-weight','.05','--quiet-only','--initial-checkpoint',str(OLD),'--learning-rate','.0001','--rank-weight','4','--rank-margin-cp','5','--hard-pair-weight','3','--protected-pair-weight','3','--score-weight','0','--anchor-weight','.1','--selection','ranking','--epochs','30','--patience','8'])
            models={'old5':NeuralEvaluator(OLD,.05,True,incremental=True,fast_features=True),'trained5':NeuralEvaluator(CP,.05,True,incremental=True,fast_features=True)}
            metrics={n:{s:pair_metrics(readrows(NEW/'pairs'/f'{s}.jsonl'),m) for s in ('train','val','test')} for n,m in models.items()}
            training=json.loads((ART/'training.json').read_text())
            write(ART/'frozen-selection.json',{'checkpoint_sha256':digest(CP),'metrics':metrics,'best_epoch':training['best_epoch'],'protocol':protocol,'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'engine/search.py',ROOT/'engine/evaluation.py',ROOT/'ml/evaluator.py',ROOT/'ml/incremental.py',ROOT/'ml/model.py',ROOT/'benchmarks/nn_baseline_v20/evaluator.py',ROOT/'benchmarks/nn_baseline_v20/incremental.py']},'old_checkpoint_sha256':digest(OLD)})
            if training['best_epoch']==0:
                phase('completed',reason='Validation retained unchanged initial weights; no redundant strength claim',matches={},app_promoted=False);return
            opening='benchmarks/openings-generalization-v21-pilot.json'
            run('fresh-strength-starts',['-m','benchmarks.generate_strength_openings','--output',opening,'--pairs','10','--seed','212500','--exclude-data',str(DATA)])
            common=['-m','ml.compare','--checkpoint',str(CP),'--nn-weight','.05','--quiet-only','--incremental','--fast-features','--opponent-checkpoint',str(OLD),'--opponent-quiet-only','--opponent-frozen-inference','--time-ms','250','--max-plies','1000']
            out=ART/'vs-heuristic';run('heuristic-pilot',common+['--opponent-nn-weight','0','--opponent-incremental','--openings',opening,'--pairs','10','--output',str(out)])
            report=json.loads((out/'report.json').read_text());matches={'pilot':report['summary']};extend=matches['pilot']['completed']==20 and not matches['pilot']['errors'] and matches['pilot']['score_fraction_completed']>=.6
            if extend:
                opening2='benchmarks/openings-generalization-v21-extension.json'
                run('freeze-confirmation',['-m','benchmarks.generate_strength_openings','--output',opening2,'--pairs','40','--seed','212501','--exclude-data',str(DATA)])
                out2=ART/'vs-heuristic-extension';run('heuristic-confirmation',common+['--opponent-nn-weight','0','--opponent-incremental','--openings',opening2,'--pairs','40','--output',str(out2)])
                extra=json.loads((out2/'report.json').read_text());matches['extension']=extra['summary']
                for g in extra['games']:g['pair']+=10
                matches['combined100']=summarize(report['games']+extra['games'])
                out3=ART/'vs-old-NN';run('old-NN-secondary',common+['--opponent-nn-weight','.25','--openings',opening,'--pairs','10','--output',str(out3)])
                matches['oldNN']=json.loads((out3/'report.json').read_text())['summary']
            phase('completed',matches=matches,extended_to_100=extend,app_promoted=False)
    except Exception as exc:phase('failed',error=str(exc));raise
if __name__=='__main__':main()
