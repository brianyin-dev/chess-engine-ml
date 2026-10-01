"""Paired inference/search speed test with score and search-tree parity."""
import json
from statistics import median
from time import perf_counter
import chess
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.nn_baseline_v20.evaluator import NeuralEvaluator as FrozenEvaluator
from engine.search import search
from ml.evaluator import NeuralEvaluator
from ml.profile_leaf_speed_v20 import ART,OLD


def main():
    output=ART/'profile-after.json'
    if output.exists():raise FileExistsError('Preserve comparison')
    records=[json.loads(l) for l in (ART/'leaves.jsonl').read_text().splitlines()]
    boards=[chess.Board(r['fen']) for r in records]
    old=FrozenEvaluator(OLD,.05,True,incremental=True)
    fast=NeuralEvaluator(OLD,.05,True,incremental=True,fast_features=True)
    mismatch=[]
    for board in boards:
        a,b=old.evaluate_position(board),fast.evaluate_position(board)
        if a!=b:mismatch.append({'fen':board.fen(),'old':a,'fast':b})
    if mismatch:raise ValueError(str(mismatch[:5]))
    rounds={'old':[],'fast':[]}
    for i in range(9):
        for name,model in ([('old',old),('fast',fast)] if i%2==0 else [('fast',fast),('old',old)]):
            start=perf_counter()
            for b in boards:model.evaluate_position(b)
            rounds[name].append((perf_counter()-start)/len(boards))
    roots=json.loads((ART/'roots.json').read_text());results=[]
    for i in range(12):
        row=roots[i*len(roots)//12];board=chess.Board(row['initial_fen'])
        for uci in row['history']:board.push_uci(uci)
        a=search(board.copy(stack=True),depth=3,eval_fn=old);b=search(board.copy(stack=True),depth=3,eval_fn=fast)
        assert (a.move,a.score,a.depth,a.nodes,a.qnodes)==(b.move,b.score,b.depth,b.nodes,b.qnodes)
        times={'old':[],'fast':[]}
        for repetition in range(3):
            for name,model in ([('old',old),('fast',fast)] if repetition%2==0 else [('fast',fast),('old',old)]):
                probe=search(board.copy(stack=True),depth=3,eval_fn=model)
                times[name].append(probe.elapsed)
        results.append({'fen':board.fen(),'move':a.move.uci(),'score':a.score,'depth':a.depth,'nodes':a.nodes,'qnodes':a.qnodes,
            'old_seconds':median(times['old']),'fast_seconds':median(times['fast'])})
    old_time=median(rounds['old']);fast_time=median(rounds['fast'])
    old_search=sum(r['old_seconds'] for r in results);fast_search=sum(r['fast_seconds'] for r in results)
    report={'policy':'Frozen b962422 inference versus opt-in fast features, same v8 weights/5% quiet correction. Alternating measurement order; sampled FENs preserve sequence. Fixeddepth3 search has no clocks/node caps. No score, move, depth or visited-node changes accepted.',
        'positions':len(boards),'integer_score_mismatches':len(mismatch),'median_microseconds':{n:median(r)*1e6 for n,r in rounds.items()},
        'round_seconds_per_position':rounds,'evaluation_time_reduction_fraction':1-fast_time/old_time,
        'fixed_depth_searches':results,'fixed_depth_search_time_reduction_fraction':1-fast_search/old_search}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    with exclusive_cpu('v20 parity and paired profiling'):main()
