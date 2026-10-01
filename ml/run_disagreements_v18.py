"""Train conservatively on actual search disagreements and require unseen gains."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize
from ml.evaluate_blends import pair_metrics
from ml.evaluator import NeuralEvaluator
from ml.generate_search_data import key

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'ml/artifacts/disagreements-v18'
NEW = ROOT / 'ml/data/disagreements-v18-2026'
DATA = ROOT / 'ml/data/aligned-v18-2026'
BASE = ROOT / 'ml/data/aligned-v17-2026'
OLD = 'ml/artifacts/quiet-ranking-v8.pt'
CP = 'ml/artifacts/disagreements-v18/trained.pt'


def prepare():
    DATA.mkdir(); (DATA / 'pairs').mkdir()
    for split in ('train', 'val', 'test'):
        sources = [BASE, NEW] if split == 'train' else [NEW]
        for relative, fields in [(f'{split}.jsonl', ('fen',)), (f'pairs/{split}.jsonl', ('good_fen', 'bad_fen'))]:
            records = {}
            for source in sources:
                for line in (source / relative).read_text().splitlines():
                    row = json.loads(line)
                    records[tuple(key(row[f]) for f in fields)] = row
            (DATA / relative).write_text(''.join(json.dumps(row)+'\n' for row in records.values()))
    manifest = {'policy':'Retain prior training labels/rankings. Fresh whole-game disagreement validation/test exclusively; prior validation/test excluded from training. Candidate selection uses fresh validation only; test consulted once for qualification.',
                'source_manifests':{str(p):hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest() for p in (BASE,NEW)}}
    for path in (DATA/'manifest.json',DATA/'pairs/manifest.json'):
        path.write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    ART.mkdir(parents=True,exist_ok=True)
    if (ART/'status.json').exists(): raise FileExistsError('Preserve existing experiment')
    def status(phase,**extra):
        tmp=ART/'status.tmp';tmp.write_text(json.dumps({'phase':phase,**extra},indent=2)+'\n');tmp.replace(ART/'status.json');print(phase,flush=True)
    def run(phase,args):
        status(phase,command=args);subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    protocol={'collection':'120 independent common book starts. Teacher self-play48plies after prefix. At each8th ply sample3 quiet real search leaves; compare equal600-node depth3 heuristic0 vsNN25. Screen teacher100ms/confirm400ms, settle12knodes. Train/val/test whole-game splits4/1/1. Exclude all previous data aliases. RetainNN-harm examples and reject unattainable margins only in training.',
              'training':'Single unchanged v8 architecture candidate: .25 quiet gate, rounded full scores; lr.0001, rank4, anchor2, protected4, hard2, score.02, margin5,20epochs/patience5. Retain prior training labels; new validation selects epoch, new test never selects it.',
              'qualification':'At least30 fresh pairs each val/test; strictly better validation ranking thanv8, no extra heuristic-correct regressions; test accuracy/regressions not worse. Failure skips games, no tuning on test.',
              'games':'Eligible candidate:20 paired games vsoptimized0 then20 vsunchanged old nonincremental25, same fresh starts250ms/depth8/max1000plies. Extend vs0 to100total if>=60% vs0 and>50% vsold, allcomplete/errorfree. No externalStockfish games or apppromotion.'}
    (ART/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    try:
        with exclusive_cpu('v18 search-disagreement training and isolated games'):
            run('tests',['-m','unittest','discover','-s','tests','-q'])
            run('freeze-collection-starts',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-disagreements-v18.json','--pairs','120','--seed','180001','--exclude-data',str(BASE)])
            run('collect-confirmed-search-disagreements',['-m','ml.generate_disagreements_v18'])
            initial_manifest = json.loads((NEW/'manifest.json').read_text())
            (ART/'collection-pilot-manifest.json').write_text(json.dumps(initial_manifest,indent=2)+'\n')
            if min(initial_manifest['counts'][s]['pairs'] for s in ('val','test')) < 30:
                (ART/'collection-extension-protocol.json').write_text(json.dumps({'reason':'Initial holdout below30pairs/split; fixed360more starts before training. Include confirmed50–99cp mistakes, separately count>=100cp; probe1500nodes. No candidate results observed.', 'training_started':False},indent=2)+'\n')
                run('freeze-supplemental-starts',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-disagreements-v18-extension.json','--pairs','360','--seed','180002','--exclude-data',str(BASE)])
                run('supplemental-disagreements',['-m','ml.generate_disagreements_v18','--append','--openings-file','benchmarks/openings-disagreements-v18-extension.json','--game-offset','120','--confirmation-threshold-cp','50','--probe-nodes','1500'])
            (ART/'teacher-collection-manifest.json').write_text((NEW/'manifest.json').read_text())
            run('freeze-student-starts',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-disagreements-v18-student.json','--pairs','120','--seed','180003','--exclude-data',str(BASE)])
            offset = sum(batch['games'] for batch in json.loads((NEW/'manifest.json').read_text()).get('batches',[{'games':120}]))
            run('student-selfplay-disagreements',['-m','ml.generate_disagreements_v18','--append','--openings-file','benchmarks/openings-disagreements-v18-student.json','--game-offset',str(offset),'--confirmation-threshold-cp','50','--probe-nodes','1500','--play-policy','student','--game-plies','96'])
            run('refine-endpoint-labels',['-m','ml.refine_disagreement_labels_v18'])
            status('prepare-retained-training');prepare()
            run('conservative-training',['-m','ml.train','--data',str(DATA),'--pairs',str(DATA/'pairs'),
                '--checkpoint',CP,'--metrics',str(ART/'training.json'),'--features','relationships','--target','residual',
                '--color-consistent','--correction-limit-cp','250','--correction-weight','.25','--quiet-only',
                '--initial-checkpoint',OLD,'--learning-rate','.0001','--rank-weight','4','--rank-margin-cp','5',
                '--hard-pair-weight','2','--protected-pair-weight','4','--score-weight','.02','--anchor-weight','2',
                '--selection','ranking','--epochs','20','--patience','5'])
            status('unseen-position-validation')
            cases={s:[json.loads(l) for l in (NEW/'pairs'/f'{s}.jsonl').read_text().splitlines()] for s in ('val','test')}
            models={'heuristic':NeuralEvaluator(OLD,0,True,incremental=True),'old':NeuralEvaluator(OLD,.25,True,incremental=True),'candidate':NeuralEvaluator(CP,.25,True,incremental=True)}
            metrics={name:{s:pair_metrics(rows,model) for s,rows in cases.items()} for name,model in models.items()}
            old,new=metrics['old'],metrics['candidate']
            eligible=(min(len(r) for r in cases.values())>=30 and new['val']['accuracy']>old['val']['accuracy']
                      and new['val']['previously_correct_regressions']<=old['val']['previously_correct_regressions']
                      and new['test']['accuracy']>=old['test']['accuracy']
                      and new['test']['previously_correct_regressions']<=old['test']['previously_correct_regressions'])
            validation={'metrics':metrics,'eligible_for_games':eligible,'policy':protocol['qualification'],
                        'checkpoint_sha256':hashlib.sha256((ROOT/CP).read_bytes()).hexdigest(),
                        'manifest_sha256':hashlib.sha256((NEW/'manifest.json').read_bytes()).hexdigest()}
            (ART/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
            if not eligible:
                status('completed',validation=validation,games_skipped='Unseen-position gate failed',app_promoted=False);return
            run('freeze-game-starts',['-m','benchmarks.generate_strength_openings','--output','benchmarks/openings-disagreements-v18-games.json','--pairs','50','--seed','182500','--exclude-data',str(DATA)])
            starts=json.loads((ROOT/'benchmarks/openings-disagreements-v18-games.json').read_text())
            for name,items in [('pilot',starts[:10]),('extension',starts[10:])]:
                (ROOT/f'benchmarks/openings-disagreements-v18-games-{name}.json').write_text(json.dumps(items,indent=2)+'\n')
            common=['-m','ml.compare','--checkpoint',CP,'--nn-weight','.25','--quiet-only','--incremental','--opponent-checkpoint',OLD,
                    '--opponent-quiet-only','--openings','benchmarks/openings-disagreements-v18-games-pilot.json','--pairs','10','--time-ms','250','--max-plies','1000']
            results={}
            for name,extra in [('vs-optimized-zero',['--opponent-nn-weight','0','--opponent-incremental']),('vs-old-NN',['--opponent-nn-weight','.25'])]:
                output=ART/name;run(name,[*common,*extra,'--output',str(output)]);results[name]=json.loads((output/'report.json').read_text())['summary']
            extend=(all(s['completed']==20 and not s['errors'] for s in results.values()) and results['vs-optimized-zero']['score_fraction_completed']>=.6 and results['vs-old-NN']['score_fraction_completed']>.5)
            if extend:
                output=ART/'vs-optimized-zero-extension'
                run('extend-to-100',['-m','ml.compare','--checkpoint',CP,'--nn-weight','.25','--quiet-only','--incremental','--opponent-checkpoint',OLD,'--opponent-nn-weight','0','--opponent-quiet-only','--opponent-incremental','--openings','benchmarks/openings-disagreements-v18-games-extension.json','--pairs','40','--time-ms','250','--max-plies','1000','--output',str(output)])
                initial=json.loads((ART/'vs-optimized-zero/report.json').read_text())['games'];extra=json.loads((output/'report.json').read_text())['games']
                for game in extra:game['pair']+=10
                results['vs-optimized-zero-100']=summarize(initial+extra)
            status('completed',validation=validation,matches=results,extended_to_100=extend,app_promoted=False)
    except Exception as exc:
        status('failed',error=str(exc));raise


if __name__=='__main__':main()
