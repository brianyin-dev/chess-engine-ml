"""Compare frozen pre-round and current searches on known diagnostic mistakes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import chess
import chess.engine
from benchmarks.analyze import replay, review
from benchmarks.pre_round.search import search as previous_search
from engine.search import search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stockfish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output file')
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / 'benchmarks/diagnostic-mistakes.json').read_text())['positions']
    sources = ['engine/search.py', 'engine/evaluation.py', 'benchmarks/pre_round/search.py',
               'benchmarks/pre_round/evaluation.py', 'benchmarks/round_probe.py',
               'benchmarks/analyze.py', 'benchmarks/diagnostic-mistakes.json']
    report = {'python':sys.version, 'python_chess':chess.__version__,
              'sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
              'policy':'Diagnostic set only. Equal thinking time; no opening holdout data. Reference scores are finite-search estimates.',
              'results':[], 'status':'running'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp=args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n')
        tmp.replace(args.output)
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve())) as engine:
        engine.configure({'Threads':1,'Hash':32})
        report['reference']={'id':engine.id,'binary_sha256':hashlib.sha256(args.stockfish.read_bytes()).hexdigest(),
                             'threads':1,'hash_mb':32,'analysis_seconds':.5}
        for case in cases:
            for budget in [.25,1.5]:
                board = replay(case)
                for label, fn in [('previous',previous_search),('current',search)]:
                    result=fn(board,depth=64,time_limit=budget)
                    stats=asdict(result)
                    stats.pop('move')
                    row={'game':case['game'],'opening':case['opening'],'fen':board.fen(),
                         'engine':label,'budget':budget,'move':result.move.uci(),'san':board.san(result.move),
                         **stats, 'reference':review(engine,board,result.move,.5)}
                    report['results'].append(row)
                    print(case['opening'],budget,label,row['san'],'depth',result.depth,
                          'loss',row['reference']['cp_loss'],flush=True)
                    save()
    report['status']='completed'
    save()


if __name__=='__main__':
    main()
