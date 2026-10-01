"""Normalize the in-flight legacy student batch before any candidate training."""
import json
from pathlib import Path
from ml.generate_disagreements_v18 import source_split

DATA=Path('ml/data/disagreements-v18-2026')
ART=Path('ml/artifacts/disagreements-v18')


def main():
    if (ART/'trained.pt').exists(): raise ValueError('Cannot change split after training')
    if (ART/'color-split-audit.json').exists(): raise FileExistsError('Normalize once')
    old=json.loads((DATA/'manifest.json').read_text())
    audit={'policy':'Swap val/test source games in odd six-game student groups to balance NN source colors before training. Training membership unchanged; teacher batches unchanged.','before':old['counts']}
    for relative in ('{split}.jsonl','pairs/{split}.jsonl'):
        groups={split:[] for split in ('train','val','test')}
        for previous in groups:
            for line in (DATA/relative.format(split=previous)).read_text().splitlines():
                row=json.loads(line);index=row['game_id']-6000000
                desired=source_split(index,index>=480)
                assert (previous=='train') == (desired=='train')
                groups[desired].append(row)
        for split,rows in groups.items():
            (DATA/relative.format(split=split)).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    old['counts']={s:{'rows':len((DATA/f'{s}.jsonl').read_text().splitlines()),'pairs':len((DATA/'pairs'/f'{s}.jsonl').read_text().splitlines())} for s in old['counts']}
    old['policy']+=' Student validation/test colors alternate by six-game groups; legacy source split normalized before training.'
    audit['after']=old['counts']
    for path in (DATA/'manifest.json',DATA/'pairs/manifest.json'):path.write_text(json.dumps(old,indent=2)+'\n')
    (ART/'color-split-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
