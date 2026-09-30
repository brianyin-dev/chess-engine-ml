"""Retain prior supervision while adding split-preserving targeted rankings."""
import argparse
import hashlib
import json
from pathlib import Path
from ml.generate_search_data import key
import chess
from ml.evaluator import NeuralEvaluator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output must be new')
    args.output.mkdir(parents=True); (args.output/'pairs').mkdir()
    reference = NeuralEvaluator('ml/artifacts/quiet-ranking-v8.pt', .25, quiet_only=True)
    counts={}
    for split in ('train','val','test'):
        rows={};pairs={}
        for source in (args.old,args.new):
            for line in (source/f'{split}.jsonl').read_text().splitlines():
                row=json.loads(line);rows[key(row['fen'])]=row
            for line in (source/'pairs'/f'{split}.jsonl').read_text().splitlines():
                row=json.loads(line);pairs[(key(row['good_fen']),key(row['bad_fen']),row['sign'])]=row
        if split == 'train':
            for row in pairs.values():
                gap = row['sign'] * (reference(chess.Board(row['good_fen'])) - reference(chess.Board(row['bad_fen'])))
                consequential = row.get('cp_loss', 0) >= 100
                row['training_weight'] = (4 if consequential and row.get('source') == 'confirmed_failure'
                                          else 2 if consequential and gap <= 0 else 1)
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows.values()))
        (args.output/'pairs'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs.values()))
        counts[split]={'score_rows':len(rows),'pairs':len(pairs),
                       'weighted_training_pairs':sum(r.get('training_weight',1)>1 for r in pairs.values()),
                       'confirmed_failure_pairs':sum(r.get('source')=='confirmed_failure' for r in pairs.values())}
    manifest={'policy':'Canonical-FEN deduplication within each split; new labels win ties. '
                        'Prior rankings retained. Training only: 4x confirmed >=100cp failures; '
                        '2x remaining >=100cp v8 ranking mistakes. Validation/test unweighted. '
                        'Trainer enforces cross-split position and whole-game separation.',
              'sources':{str(path):hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
                         for path in (args.old,args.new)},'counts':counts}
    for path in (args.output/'manifest.json',args.output/'pairs'/'manifest.json'):
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
