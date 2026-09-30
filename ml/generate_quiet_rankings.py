"""Split-isolated ranking supervision after teacher-guided tactical settling."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import chess
import chess.engine
from engine.evaluation import evaluate
from engine.search import search
from ml.dataset import load_rows
from ml.evaluator import NeuralEvaluator
from ml.generate_search_data import key


def is_tactical(board, move):
    return board.is_check() or board.is_capture(move) or bool(move.promotion)


def score(info):
    return max(-1500, min(1500, info['score'].white().score(mate_score=1500)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--failures',type=Path,required=True)
    p.add_argument('--stockfish',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-pairs',type=int,default=2000)
    p.add_argument('--heldout-pairs',type=int,default=400)
    args = p.parse_args()
    if args.output.exists() or min(args.train_pairs,args.heldout_pairs)<1:
        p.error('fresh output and positive quotas required')
    started=time.monotonic()
    rng=random.Random(829)
    base={s:load_rows(args.data/f'{s}.jsonl') for s in ('train','val','test')}
    owners={key(r['fen']):s for s,rows in base.items() for r in rows}
    for split in base:
        for r in [json.loads(l) for l in (args.data/'pairs'/f'{split}.jsonl').read_text().splitlines()]:
            for name in ('good_fen','bad_fen'):
                k=key(r[name])
                if k in owners and owners[k]!=split:
                    raise ValueError('source pair alias crosses splits')
                owners[k]=split
    model=NeuralEvaluator(args.checkpoint)
    args.output.mkdir(parents=True)
    pairdir=args.output/'pairs';pairdir.mkdir()
    cache={};stats={'requested_nodes':0,'cached_analyses':0}
    counts={}
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve())) as sf:
        sf.configure({'Threads':1,'Hash':64})
        def analyse(board,nodes=1200):
            k=(board.fen(),nodes)
            if k in cache:
                stats['cached_analyses']+=1
                return cache[k]
            info=sf.analyse(board,chess.engine.Limit(nodes=nodes),game=object())
            stats['requested_nodes']+=nodes
            cache[k]=info
            return info
        def settle(board,nodes):
            board=board.copy(stack=False)
            line=[]
            for _ in range(17):
                if board.is_game_over():
                    return None
                info=analyse(board,nodes)
                pv=info.get('pv',[])
                if not pv:
                    return None
                move=pv[0]
                if not is_tactical(board,move):
                    return board,score(info),line
                line.append(move.uci());board.push(move)
            return None
        for split,source in base.items():
            target=args.train_pairs if split=='train' else args.heldout_pairs
            rows={key(r['fen']):r for r in source if not chess.Board(r['fen']).is_check() and
                  not next(chess.Board(r['fen']).generate_legal_captures(),None) and
                  not chess.Board(r['fen']).is_game_over()}
            initial_rows=len(rows)
            pairs=[];seen=set();refined=0;roots=0;hard=0;settled_plies=0
            eligible=[r for r in source if not chess.Board(r['fen']).is_game_over()]
            endings=[r for r in eligible if len(chess.Board(r['fen']).piece_map())<=12]
            known=[]
            if split=='train':
                failures=json.loads(args.failures.read_text())
                for i,g in enumerate(failures['games']):
                    failure=g.get('first_costly_move')
                    if failure:
                        known.append({'fen':failure['fen'],'game_id':1000000+i,'ply':chess.Board(failure['fen']).ply(),
                                      'bad_move':failure['played'],'source':'confirmed_failure'})
            def compare(root,board,good_move,bad_move,deep=False):
                nonlocal hard,settled_plies
                if len(pairs)>=target or good_move==bad_move:
                    return
                a,b=board.copy(),board.copy();a.push(good_move);b.push(bad_move)
                first,second=settle(a,4800 if deep else 1200),settle(b,4800 if deep else 1200)
                if not first or not second:
                    return
                good,good_cp,good_line=first;bad,bad_cp,bad_line=second
                sign=1 if board.turn else -1
                gap=sign*(good_cp-bad_cp)
                if gap<50 or abs(good_cp)>=1500 or abs(bad_cp)>=1500:
                    return
                keys=key(good.fen()),key(bad.fen())
                if keys[0]==keys[1] or keys in seen or any(owners.get(k,split)!=split for k in keys):
                    return
                seen.add(keys)
                base_gap=sign*(evaluate(good)-evaluate(bad));hard+=base_gap<=0
                settled_plies+=len(good_line)+len(bad_line)
                pairs.append({'good_fen':good.fen(),'bad_fen':bad.fen(),'sign':sign,
                              'game_id':root['game_id'],'cp_loss':gap,'heuristic_gap_cp':base_gap,
                              'root_fen':board.fen(),'good_move':good_move.uci(),'bad_move':bad_move.uci(),
                              'good_tactical_line':good_line,'bad_tactical_line':bad_line,
                              'label_nodes_per_settling_step':4800 if deep else 1200,
                              'source':root.get('source','sampled_quiet_ranking')})
                for k,leaf,cp in zip(keys,(good,bad),(good_cp,bad_cp)):
                    owners[k]=split
                    if k not in rows:
                        rows[k]={'fen':leaf.fen(),'score_cp':cp,'game_id':root['game_id'],
                                 'ply':leaf.ply(),'source':'teacher_settled_quiet','root_fen':board.fen()}
            while len(pairs)<target:
                roots+=1
                if roots>target*20:
                    raise RuntimeError('could not reach ranking quota within root budget')
                root=known.pop(0) if known else rng.choice(endings if roots%4==0 and endings else eligible)
                board=chess.Board(root['fen'])
                heuristic=search(board,depth=2,node_limit=250)
                neural=search(board,depth=2,node_limit=250,eval_fn=model)
                disagreement=heuristic.move!=neural.move
                deep=(disagreement or root.get('bad_move')) and refined<target//4
                if deep:refined+=1
                infos=sf.analyse(board,chess.engine.Limit(nodes=6000 if deep else 2000),multipv=2,game=object())
                stats['requested_nodes']+=6000 if deep else 2000
                if not infos or not infos[0].get('pv'):
                    continue
                best=infos[0]['pv'][0]
                candidates=[neural.move,heuristic.move]
                if root.get('bad_move'):candidates.insert(0,chess.Move.from_uci(root['bad_move']))
                if len(infos)>1 and infos[1].get('pv'):candidates.append(infos[1]['pv'][0])
                # A diverse inferior candidate provides anchors when both engines agree.
                candidates.append(rng.choice(list(board.legal_moves)))
                for bad in dict.fromkeys(candidates):
                    compare(root,board,best,bad,deep)
                if roots%50==0:
                    print(f'{split}: {len(pairs)}/{target} pairs, {hard} heuristic-wrong, {refined} refined roots',flush=True)
            (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows.values()))
            (pairdir/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
            counts[split]={'pairs':len(pairs),'score_rows':len(rows),'reused_score_rows':initial_rows,
                           'roots':roots,'refined_roots':refined,'heuristic_wrong_pairs':hard,
                           'tactical_plies_settled':settled_plies}
            print(split,counts[split],flush=True)
    manifest={'seed':829,'counts':counts,**stats,'elapsed_seconds':time.monotonic()-started,
              'policy':'Whole-game source splits; canonical FEN/color-mirror aliases excluded across splits including pairs. Settling follows teacher-selected captures, promotions and check evasions until its best continuation is quiet. Other legal captures may remain. No terminal/mate-saturated endpoint pairs. Existing capture-free score labels reused.',
              'extra_compute_policy':'Up to quarter-quota roots refined when shallow heuristic and NN moves disagree or are confirmed failures. Quotas refer to pairs; some roots yield several pairs.',
              'source_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
              'failures_sha256':hashlib.sha256(args.failures.read_bytes()).hexdigest(),
              'student_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'stockfish_sha256':hashlib.sha256(args.stockfish.read_bytes()).hexdigest()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (pairdir/'manifest.json').write_text(json.dumps({'counts':counts,'parent_sha256':hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':
    main()
