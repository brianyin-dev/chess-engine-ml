"""Refine fixed endpoints before training; never use candidate predictions to filter."""
import json
from pathlib import Path
import chess
import chess.engine

DATA=Path('ml/data/disagreements-v18-2026')
ART=Path('ml/artifacts/disagreements-v18')
SF=Path('tools/stockfish-sf19/stockfish/stockfish-macos-universal')


def main():
    if (ART/'trained.pt').exists(): raise ValueError('Labels must freeze before training')
    if (ART/'label-refinement.json').exists(): raise FileExistsError('Refine once')
    pairs={s:[json.loads(l) for l in (DATA/'pairs'/f'{s}.jsonl').read_text().splitlines()] for s in ('train','val','test')}
    fens=sorted({r[f] for records in pairs.values() for r in records for f in ('good_fen','bad_fen')})
    scores={}
    with chess.engine.SimpleEngine.popen_uci(str(SF.resolve())) as sf:
        sf.configure({'Threads':1,'Hash':32})
        for i,fen in enumerate(fens):
            exact=None
            with sf.analysis(chess.Board(fen),chess.engine.Limit(nodes=64000),game=object()) as analysis:
                for info in analysis:
                    if 'score' in info and info.get('pv') and not info.get('lowerbound') and not info.get('upperbound'):
                        exact=info['score'].white()
            if exact is None: raise ValueError('No completed exact score')
            cp=exact.score();scores[fen]=cp if cp is not None and abs(cp)<1500 else None
            if (i+1)%50==0:print(f'Refined {i+1}/{len(fens)} endpoints',flush=True)
    audit={'policy':'Fixed existing endpoints reanalyzed at64knodes, retaining last completed exact scored PV; no candidate predictions. Drop mate/saturated labels or settled score gaps below50cp. Root400ms confirmations unchanged. Freeze before training.', 'endpoints':len(fens),'counts':{}}
    for split,records in pairs.items():
        kept=[];retained_fens=set()
        for row in records:
            a,b=scores[row['good_fen']],scores[row['bad_fen']]
            if a is None or b is None or row['sign']*(a-b)<50: continue
            row['initial_settled_gap_cp']=row['cp_loss'];row['cp_loss']=row['sign']*(a-b);row['endpoint_label_nodes']=64000
            kept.append(row);retained_fens.update([row['good_fen'],row['bad_fen']])
        rows=[json.loads(l) for l in (DATA/f'{split}.jsonl').read_text().splitlines()]
        rows=[{**row,'score_cp':scores[row['fen']]} for row in rows if row['fen'] in retained_fens]
        (DATA/'pairs'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in kept))
        (DATA/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        audit['counts'][split]={'before_pairs':len(records),'pairs':len(kept),'rows':len(rows)}
    manifest=json.loads((DATA/'manifest.json').read_text());manifest['endpoint_refinement']=audit
    manifest['counts']={s:{'rows':r['rows'],'pairs':r['pairs']} for s,r in audit['counts'].items()}
    for p in (DATA/'manifest.json',DATA/'pairs/manifest.json'):p.write_text(json.dumps(manifest,indent=2)+'\n')
    (ART/'label-refinement.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
