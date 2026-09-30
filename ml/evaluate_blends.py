"""Predeclared correction-weight comparisons on split-isolated settled pairs."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import chess
from engine.evaluation import evaluate
from ml.evaluator import NeuralEvaluator


def pair_metrics(records,evaluator):
    gaps=[r['sign']*(evaluator(chess.Board(r['good_fen']))-evaluator(chess.Board(r['bad_fen']))) for r in records]
    hard=[i for i,r in enumerate(records) if r['heuristic_gap_cp']<=0]
    easy=[i for i,r in enumerate(records) if r['heuristic_gap_cp']>0]
    return {'pairs':len(records),'correct':sum(v>0 for v in gaps),
            'accuracy':sum(v>0 for v in gaps)/len(records),
            'hard_pairs':len(hard),'hard_correct':sum(gaps[i]>0 for i in hard),
            'previously_correct_regressions':sum(gaps[i]<=0 for i in easy)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output must be new')
    records={s:[json.loads(l) for l in (args.data/'pairs'/f'{s}.jsonl').read_text().splitlines()] for s in ('val','test')}
    report={'policy':'Fixed weights and gates evaluated. Gate selected per weight using validation ranking only; ties prefer gated. Test never selects a configuration. All test comparisons reported.',
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
            'heuristic':{s:pair_metrics(rows,evaluate) for s,rows in records.items()},
            'previous_full':{s:pair_metrics(rows,NeuralEvaluator(args.previous)) for s,rows in records.items()},
            'configurations':{},'selected_for_games':[]}
    for weight in (0,.1,.25,.5,1):
        for gate in (False,True):
            name=f'w{weight:g}-'+('quiet' if gate else 'all')
            e=NeuralEvaluator(args.checkpoint,weight,gate)
            report['configurations'][name]={'weight':weight,'quiet_only':gate,
                **{s:pair_metrics(rows,e) for s,rows in records.items()}}
    for weight in (.1,.25,.5):
        options=[(name,c) for name,c in report['configurations'].items() if c['weight']==weight]
        name,c=max(options,key=lambda item:(item[1]['val']['accuracy'],item[1]['quiet_only']))
        report['selected_for_games'].append({'name':name,'weight':weight,'quiet_only':c['quiet_only']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
