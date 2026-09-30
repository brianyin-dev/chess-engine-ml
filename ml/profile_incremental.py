"""Verify unchanged NN decisions and measure complete-search savings."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import opening_board
from engine.search import search
from ml.evaluator import NeuralEvaluator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--openings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.output.exists() or args.repeats<1:p.error('new output and positive repetitions required')
    rows=[]
    for opening in json.loads(args.openings.read_text()):
        b=opening_board(opening);trials={'reference':[],'incremental':[]};expected=None
        for repeat in range(args.repeats):
            for name in (list(trials) if repeat%2 else list(reversed(trials))):
                e=NeuralEvaluator(args.checkpoint,.25,quiet_only=True,incremental=name=='incremental')
                r=search(b,depth=3,eval_fn=e);output=(r.move.uci(),r.score,r.nodes,r.qnodes)
                if expected is None:expected=output
                assert output==expected,(b.fen(),name,output,expected)
                trials[name].append(r.elapsed)
        rows.append({'fen':b.fen(),'output':expected,'trials_seconds':trials})
        print(f'Profile {len(rows)}',flush=True)
    report={'positions':rows,'median_total_seconds':{k:sum(median(row['trials_seconds'][k]) for row in rows) for k in trials},
            'score_move_node_mismatches':0,'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'openings_sha256':hashlib.sha256(args.openings.read_bytes()).hexdigest(),
            'policy':'Alternating depth3 complete searches with unchanged v8 weights and .25 quiet gate; fresh evaluator per trial. Same moves, scores, nodes, qnodes required.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(report['median_total_seconds'])


if __name__=='__main__':
    with exclusive_cpu('incremental hybrid search profile'):main()
