"""Separate static-evaluation quality from search speed on development holdouts."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from statistics import mean,median
from time import perf_counter
import chess
import chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from engine.evaluation import evaluate_position
from engine.search import search
from ml.evaluator import NeuralEvaluator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--stockfish',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--positions',type=int,default=32)
    p.add_argument('--failures',type=Path,help='Diagnose confirmed first mistakes from actual lost/drawn games instead of random holdouts')
    args=p.parse_args()
    if args.output.exists():p.error('output must be new')
    rng=random.Random(14250)
    boards=[chess.Board(json.loads(l)['fen']) for l in (args.data/'test.jsonl').read_text().splitlines()]
    boards=[b for b in boards if not b.is_game_over()]
    samples=rng.sample(boards,min(args.positions,len(boards)))
    if args.failures:
        cases=json.loads(args.failures.read_text())['games']
        samples=[chess.Board(g['first_costly_move']['fen']) for g in cases if g['first_costly_move']]
        samples=list({b.fen():b for b in samples}.values())[:args.positions]
        if not samples:raise ValueError('no confirmed mistakes to diagnose')
    models={'old':NeuralEvaluator('ml/artifacts/quiet-ranking-v8.pt',.25,quiet_only=True),
            'compact':NeuralEvaluator(args.checkpoint,.25,quiet_only=True)}
    functions={'heuristic':evaluate_position,**{k:v.evaluate_position for k,v in models.items()}}
    trials={k:[] for k in functions}
    probe=boards[:512]+[b.mirror() for b in boards[:512]]
    for i in range(7):
        for name in (list(functions) if i%2 else list(reversed(functions))):
            start=perf_counter()
            for b in probe:functions[name](b)
            trials[name].append((perf_counter()-start)*1000/len(probe))
    rows=[]
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as teacher:
        teacher.configure({'Threads':1,'Hash':16})
        for index,board in enumerate(samples):
            row={'fen':board.fen(),'searches':{}}
            for mode in ('equal_depth','equal_time'):
                observations={}
                for name in functions:
                    result=search(board,depth=3 if mode=='equal_depth' else 8,
                                  eval_fn=None if name=='heuristic' else models[name],
                                  time_limit=None if mode=='equal_depth' else .25)
                    assert result.move in board.legal_moves
                    stats=asdict(result);stats['move']=result.move.uci()
                    observations[name]=stats
                row['searches'][mode]=observations
            moves={chess.Move.from_uci(s['move']) for v in row['searches'].values() for s in v.values()}
            scores={}
            # These are teacher diagnostics, not limited-strength opponent games.
            for move in sorted(moves,key=lambda m:m.uci()):
                info=teacher.analyse(board,chess.engine.Limit(depth=12),root_moves=[move],game=object())
                scores[move.uci()]=info['score'].pov(board.turn).score(mate_score=10000)
            best=teacher.analyse(board,chess.engine.Limit(depth=12),game=object())
            row['teacher']={'depth':12,'selected_move_cp':scores,'best_cp':best['score'].pov(board.turn).score(mate_score=10000)}
            rows.append(row)
            print(f'Diagnostic {index+1}/{len(samples)}',flush=True)
    summary={}
    for mode in ('equal_depth','equal_time'):
        summary[mode]={}
        for name in functions:
            stats=[r['searches'][mode][name] for r in rows]
            regrets=[max(0,r['teacher']['best_cp']-r['teacher']['selected_move_cp'][s['move']]) for r,s in zip(rows,stats)]
            summary[mode][name]={'mean_teacher_regret_cp':mean(regrets),'regrets_at_least_100cp':sum(v>=100 for v in regrets),
                                'mean_depth':mean(s['depth'] for s in stats),'mean_nodes':mean(s['nodes']+s['qnodes'] for s in stats),
                                'mean_seconds':mean(s['elapsed'] for s in stats)}
    report={'policy':('Confirmed first mistakes from actual games; FEN position diagnostics without repetition history; not new generalization evidence. ' if args.failures else
                     'Fixed seed sample from reused development test split; diagnostic only, not new strength evidence. ')+
                     'Same search algorithm, depth3 or250ms. Teacher depth12 restricted-root scores are approximate.',
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'old_checkpoint_sha256':hashlib.sha256(Path('ml/artifacts/quiet-ranking-v8.pt').read_bytes()).hexdigest(),
            'stockfish_sha256':hashlib.sha256(args.stockfish.read_bytes()).hexdigest(),
            'engine_sources_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path('engine').glob('*.py'))},
            'median_ms_per_evaluation':{k:median(v) for k,v in trials.items()},'timing_trials_ms':trials,'summary':summary,'positions':rows}
    if args.failures:report['failures_sha256']=hashlib.sha256(args.failures.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with exclusive_cpu('compact NN diagnostics'):main()
