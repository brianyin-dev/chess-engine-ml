"""Compare learned mistakes and rounded inference consistency without search."""
import argparse
import json
from pathlib import Path
import chess
import torch
from engine.evaluation import evaluate
from ml.evaluator import NeuralEvaluator
from ml.evaluate_blends import pair_metrics
from ml.model import board_to_tensor, SCORE_SCALE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output must be new')
    data=Path('ml/data/targeted-v10-2026')
    pairs={s:[json.loads(l) for l in (data/'pairs'/f'{s}.jsonl').read_text().splitlines()]
           for s in ('train','val','test')}
    confirmed=[r for r in pairs['train'] if r.get('source')=='confirmed_failure']
    recent=[r for r in confirmed if r['game_id']>=2000000]
    boards=[chess.Board(json.loads(l)['fen']) for l in (data/'test.jsonl').read_text().splitlines()[:512]]
    report={'policy':'Training mistake diagnostics are not generalization evidence. '
                    'Same .25 quiet weight on all comparisons; no search or latency measurements.',
            'models':{}}
    for name,path in [('v8','ml/artifacts/quiet-ranking-v8.pt'),('v10','ml/artifacts/targeted-ranking-v10.pt')]:
        e=NeuralEvaluator(path,.25,quiet_only=True)
        parity=[];symmetry=[]
        with torch.inference_mode():
            for b in boards:
                use_nn=not (b.is_check() or next(b.generate_legal_captures(),None))
                predicted=round(evaluate(b)+e.model(board_to_tensor(b,e.input_size)).item()*SCORE_SCALE*.25) if use_nn else evaluate(b)
                parity.append(abs(predicted-e(b)))
                symmetry.append(abs(e(b)+e(b.mirror())))
        report['models'][name]={'confirmed_training_pairs':pair_metrics(confirmed,e),
            'recent_confirmed_training_pairs':pair_metrics(recent,e),
            'validation_pairs':pair_metrics(pairs['val'],e),
            'test_pairs':pair_metrics(pairs['test'],e),
            'inference_parity':{'positions':len(boards),'different_integer_scores':sum(v!=0 for v in parity),
                                'max_difference_cp':max(parity)},
            'color_symmetry':{'max_error_cp':max(symmetry)}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
