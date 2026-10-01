"""One recorded25% follow-up; validation first, fresh games primary."""
import json,subprocess,sys
from ml.run_generalization_v21 import ART,DATA,NEW,ROOT,OLD,write,readrows,digest
from ml.evaluator import NeuralEvaluator
from ml.evaluate_blends import pair_metrics
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize
CP=ART/'trained25.pt'
def main():
    if CP.exists():raise FileExistsError('Preserve checkpoint')
    write(ART/'followup25-protocol.json',{'reason':'5% validation retained epochzero. One larger established25% quiet blend, fixed inference and same family-separated data, chosen from correction range and validation failure, nottest results.','training':'Same objective/hyperparameters except runtimeblend25% and initializationoldv8; bestvalidation checkpoint including epochzero.','games':'Selected25% configuration frozen before fresh10paired openings250ms; report whether learned weights retained. Primary heuristic pilot>=60% and20complete/errorfree triggers80newgames; oldNNsecondary20 only ifheurpilotpassed. No apppromotion. Ifepochzero retained, game result is old25reference replication, nottraining improvement.'})
    def run(args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    with exclusive_cpu('v21 followup25 and isolated strength'):
        run(['-m','ml.train','--data',str(DATA),'--pairs',str(DATA/'pairs'),'--checkpoint',str(CP),'--metrics',str(ART/'training25.json'),'--features','relationships','--target','residual','--color-consistent','--correction-limit-cp','250','--correction-weight','.25','--quiet-only','--initial-checkpoint',str(OLD),'--learning-rate','.0001','--rank-weight','4','--rank-margin-cp','5','--hard-pair-weight','3','--protected-pair-weight','3','--score-weight','0','--anchor-weight','.1','--selection','ranking','--epochs','30','--patience','8'])
        metrics={n:{s:pair_metrics(readrows(NEW/'pairs'/f'{s}.jsonl'),NeuralEvaluator(p,.25,True,incremental=True,fast_features=True)) for s in ('train','val','test')} for n,p in [('old25',OLD),('trained25',CP)]}
        training=json.loads((ART/'training25.json').read_text());write(ART/'frozen25.json',{'checkpoint_sha256':digest(CP),'old_checkpoint_sha256':digest(OLD),'best_epoch':training['best_epoch'],'metrics':metrics,'weight':.25,'quiet_only':True,'incremental':True,'fast_features':True})
        opening='benchmarks/openings-generalization-v21-pilot.json'
        run(['-m','benchmarks.generate_strength_openings','--output',opening,'--pairs','10','--seed','212500','--exclude-data',str(DATA)])
        common=['-m','ml.compare','--checkpoint',str(CP),'--nn-weight','.25','--quiet-only','--incremental','--fast-features','--opponent-checkpoint',str(OLD),'--opponent-quiet-only','--opponent-frozen-inference','--time-ms','250','--max-plies','1000']
        out=ART/'vs-heuristic';write(ART/'status.json',{'phase':'heuristic-pilot','weight':.25,'best_epoch':training['best_epoch']})
        run(common+['--opponent-nn-weight','0','--opponent-incremental','--openings',opening,'--pairs','10','--output',str(out)])
        report=json.loads((out/'report.json').read_text());matches={'pilot':report['summary']};extend=matches['pilot']['completed']==20 and not matches['pilot']['errors'] and matches['pilot']['score_fraction_completed']>=.6
        if extend:
            opening2='benchmarks/openings-generalization-v21-extension.json'
            run(['-m','benchmarks.generate_strength_openings','--output',opening2,'--pairs','40','--seed','212501','--exclude-data',str(DATA)])
            out2=ART/'vs-heuristic-extension';write(ART/'status.json',{'phase':'100-game-confirmation'})
            run(common+['--opponent-nn-weight','0','--opponent-incremental','--openings',opening2,'--pairs','40','--output',str(out2)])
            extra=json.loads((out2/'report.json').read_text());matches['extension']=extra['summary']
            for g in extra['games']:g['pair']+=10
            matches['combined100']=summarize(report['games']+extra['games'])
            out3=ART/'vs-old-NN';run(common+['--opponent-nn-weight','.25','--openings',opening,'--pairs','10','--output',str(out3)]);matches['oldNN']=json.loads((out3/'report.json').read_text())['summary']
        write(ART/'status.json',{'phase':'completed','matches':matches,'extended_to_100':extend,'weight':.25,'best_epoch':training['best_epoch'],'app_promoted':False})
if __name__=='__main__':main()
