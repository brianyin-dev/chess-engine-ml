"""Overfit confirmed failures to test learnability, never playing strength."""
import argparse
import hashlib
import json
from pathlib import Path
import chess
import torch
from ml.evaluator import NeuralEvaluator
from ml.model import board_to_tensor, SCORE_SCALE
from engine.evaluation import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output must be new')
    torch.manual_seed(819)
    e = NeuralEvaluator(args.checkpoint)
    model = e.model
    pairs = json.loads(args.analysis.read_text())['pairs']
    good = [chess.Board(r['good_fen']) for r in pairs]
    bad = [chess.Board(r['bad_fen']) for r in pairs]
    x = torch.stack([board_to_tensor(b,e.input_size) for b in good])
    y = torch.stack([board_to_tensor(b,e.input_size) for b in bad])
    base = torch.tensor([(evaluate(g)-evaluate(b))/SCORE_SCALE for g,b in zip(good,bad)])
    sign = torch.tensor([r['sign'] for r in pairs])
    def deltas():
        return sign*(model(x)-model(y)+base)*SCORE_SCALE
    before = deltas().detach().tolist()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for step in range(1,2001):
        optimizer.zero_grad()
        delta = deltas()
        loss = torch.relu(20-delta).mean()/SCORE_SCALE
        loss.backward()
        optimizer.step()
        if (deltas().detach() >= 20).all():
            break
    after = deltas().detach().tolist()
    args.output.mkdir(parents=True)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    saved['state_dict'] = model.state_dict()
    saved['diagnostic_only'] = True
    torch.save(saved,args.output/'memorization-only.pt')
    report = {'seed':819,
              'initial_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'analysis_sha256':hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
              'policy':'Training-set memorization diagnostic only. No held-out claim; never promoted to app.',
              'pairs':len(pairs),'steps':step,'margin_cp':20,
              'correct_before':sum(v>0 for v in before),'correct_after':sum(v>0 for v in after),
              'all_margins_reached':all(v>=20 for v in after),'before_cp':before,'after_cp':after}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
