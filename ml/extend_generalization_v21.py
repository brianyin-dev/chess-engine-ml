"""Pre-fit coverage extension; balance source splits, never teacher outcomes."""
import collections,hashlib,json,random,sys
import chess,chess.engine
from engine.opening_book import choose_book_move
from ml.run_generalization_v21 import ART,NEW,DATA,ROOT,previous,write,readrows
from benchmarks.cpu_lock import exclusive_cpu

def main():
    assert not (ART/'trained.pt').exists()
    extension=ART/'coverage-extension'
    if extension.exists():raise FileExistsError('Preserve extension')
    extension.mkdir()
    write(extension/'protocol.json',{'policy':'Before model fitting: add120train,100val,100test source games, max6newgames/family, balance family exposure by split only. Fixed hash assignment unchanged. Teacher outcomes never select sources. Label with samev20 policy; merge existing labels, globally exclude them from new labels. Collection minimum30 temporarily deferred until merged counts checked.'})
    rng=random.Random(210022);counts=collections.Counter();families=collections.Counter();records=[];games=[];seen=set()
    with exclusive_cpu('v21 pre-fit coverage extension'):
        with chess.engine.SimpleEngine.popen_uci(str(previous.SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            for attempt in range(30000):
                if counts=={'train':120,'val':100,'test':100}:break
                board=chess.Board();opening=[];familyboard=chess.Board()
                for ply in range(12):
                    choice=choose_book_move(board,ROOT/'books/gm2001.bin',rng=rng)
                    if choice is None:break
                    board.push(choice.move);opening.append(choice.move.uci())
                    if ply<4:familyboard.push(choice.move)
                if len(opening)<4:continue
                family=previous.canonical(familyboard.fen());fid=int(hashlib.sha256(family.encode()).hexdigest()[:12],16)
                bucket=fid%10;s='train' if bucket<6 else 'val' if bucket<8 else 'test'
                if counts[s]>=({'train':120,'val':100,'test':100}[s]) or families[fid]>=6:continue
                gid=240+len(games);families[fid]+=1;counts[s]+=1;snapshots=[];moves=[]
                for ply in range(72):
                    if board.is_game_over():break
                    if ply>=6 and not board.is_check() and next(board.generate_legal_captures(),None) is None:snapshots.append(board.fen())
                    move=sf.play(board,chess.engine.Limit(nodes=2000),game=gid).move;moves.append(move.uci());board.push(move)
                rng.shuffle(snapshots)
                for fen in snapshots[:8]:
                    k=previous.canonical(fen)
                    if k in seen:continue
                    seen.add(k);records.append({'fen':fen,'game_id':fid,'source_game':gid,'split':s})
                games.append({'game_id':gid,'family':family,'split':s,'opening':opening,'moves':moves})
                if len(games)%20==0:print('Coverage games',len(games),dict(counts),flush=True)
        if counts!={'train':120,'val':100,'test':100}:raise ValueError('Insufficient family coverage')
        (extension/'leaves.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));write(extension/'source-games.json',games)
        previous.ART=extension;previous.NEW=ROOT/'ml/data/generalization-v21-extension'
        # Existing new labels now count as prior, preventing duplicate aliases.
        try:previous.collect()
        except ValueError as e:
            if 'Insufficient targeted holdout' not in str(e):raise
        raw_manifest=json.loads((previous.NEW/'manifest.json').read_text())
        raw_manifest['policy']='Family-balanced pre-fit extension, same64knode settled ranking labels and exact rounded5% reachability filter; first4ply canonical families own hash split. Prior labels including initialv21 excluded.'
        write(previous.NEW/'manifest.json',raw_manifest);write(previous.NEW/'pairs/manifest.json',raw_manifest)
        for s in ('train','val','test'):
            for suffix in (f'{s}.jsonl',f'pairs/{s}.jsonl'):
                p=NEW/suffix;p.write_text(p.read_text()+(previous.NEW/suffix).read_text())
        counts={s:len(readrows(NEW/'pairs'/f'{s}.jsonl')) for s in ('train','val','test')}
        if min(counts['val'],counts['test'])<30:raise ValueError(f'Coverage still insufficient: {counts}')
        manifest={'counts':{s:{'rows':len(readrows(NEW/f'{s}.jsonl')),'pairs':counts[s],'families':len({r['game_id'] for r in readrows(NEW/'pairs'/f'{s}.jsonl')})} for s in counts},'policy':'Initial240 plus fixed320 family-balanced sourcegames; first4ply canonical families retain same hash split; v20 settled reachable5% ranking labels. No model fitted before extension.'}
        write(NEW/'manifest.json',manifest);write(NEW/'pairs/manifest.json',manifest)
        print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
