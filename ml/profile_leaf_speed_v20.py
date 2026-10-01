"""Freeze real search-leaf samples and profile unchanged inference components."""
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import random
from statistics import median
from time import perf_counter
import chess
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.nn_baseline_v20.evaluator import NeuralEvaluator
from engine.search import search

ART=Path('ml/artifacts/leaf-speed-v20')
OLD=Path('ml/artifacts/quiet-ranking-v8.pt')


class Leaves:
    cacheable_by_fen=True
    def __init__(self,model,rng):self.model=model;self.rng=rng;self.fens=[];self.calls=0;self.quiet_calls=0
    def evaluate_position(self,board):
        self.calls+=1
        if not board.is_check() and not next(board.generate_legal_captures(),None):
            self.quiet_calls+=1
            if len(self.fens)<40:self.fens.append(board.fen())
            else:
                index=self.rng.randrange(self.quiet_calls)
                if index<40:self.fens[index]=board.fen()
        return self.model.evaluate_position(board)
    __call__=evaluate_position


def main():
    ART.mkdir(parents=True,exist_ok=True)
    output=ART/'profile-before.json'
    if output.exists():raise FileExistsError('Preserve profile')
    model=NeuralEvaluator(OLD,.05,True,incremental=True);rng=random.Random(200001);roots=[];records=[]
    sources=[Path(f'ml/artifacts/balanced-v19/{name}/report.json') for name in ('vs-heuristic','vs-old-NN')]
    for source in sources:
        for game in json.loads(source.read_text())['games']:
            for fraction in (.2,.5,.8):
                board=chess.Board(game['initial_fen'])
                for move in game['opening_moves']:board.push_uci(move)
                index=int(fraction*len(game['moves']))
                for item in game['moves'][:index]:board.push_uci(item['uci'])
                if board.is_game_over():continue
                sampler=Leaves(model,rng);probe=search(board.copy(stack=True),depth=3,node_limit=6000,eval_fn=sampler)
                group=game['pair']-1;split='train' if group%5<3 else 'val' if group%5==3 else 'test'
                gid=8000000+group
                roots.append({'fen':board.fen(),'initial_fen':game['initial_fen'],'history':[m.uci() for m in board.move_stack],
                    'game_id':gid,'split':split,'source':str(source),'depth':probe.depth,'nodes':probe.nodes,'qnodes':probe.qnodes})
                records.extend({'fen':fen,'game_id':gid,'split':split,'root_index':len(roots)-1} for fen in sampler.fens)
    (ART/'leaves.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (ART/'roots.json').write_text(json.dumps(roots,indent=2)+'\n')
    boards=[chess.Board(r['fen']) for r in records]
    for b in boards:model.evaluate_position(b)
    rounds=[]
    for _ in range(7):
        start=perf_counter()
        for b in boards:model.evaluate_position(b)
        rounds.append((perf_counter()-start)/len(boards))
    profile=cProfile.Profile();profile.enable()
    for b in boards:model.evaluate_position(b)
    profile.disable();profile.dump_stats(str(ART/'before.prof'))
    stats=pstats.Stats(profile)
    top=[{'file':f[0],'line':f[1],'function':f[2],'calls':v[1],'self_seconds':v[2],'cumulative_seconds':v[3]} for f,v in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:25]]
    report={'positions':len(boards),'roots':len(roots),'median_microseconds':median(rounds)*1e6,'round_seconds_per_position':rounds,'top_functions':top,
        'policy':'Quiet evaluator calls from bounded depth3/6000-node searches of v19 development games. Sampling is not used for speed claims. Profiles use standalone search-only inference on saved FENs, preserving sequence. Same opening pair owns all variants/colors across train/val/test.',
        'checkpoint_sha256':hashlib.sha256(OLD.read_bytes()).hexdigest(),'leaves_sha256':hashlib.sha256((ART/'leaves.jsonl').read_bytes()).hexdigest()}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    with exclusive_cpu('v20 leaf collection and before profile'):main()
