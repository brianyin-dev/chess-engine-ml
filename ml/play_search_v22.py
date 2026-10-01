"""Recorded exploratory pilot after a tied-depth screen with validation gains."""
import json,subprocess,sys
from ml.run_search_v22 import ART,DATA,ROOT,CP,OLD,write,digest
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import summarize

def main():
    screen=json.loads((ART/'move-screen.json').read_text());training=json.loads((ART/'training.json').read_text())
    summary=screen['summary'];a=training['initial_validation'];b=training['selected_validation']
    assert b['correct']>a['correct'] and b['regressions_from_old']==0
    assert screen['comparable']>=16
    for baseline in ('old25',):
        assert summary['trained25']['mean_cp_regret']<=summary[baseline]['mean_cp_regret']
        assert summary['trained25']['bad_100cp_moves']<=summary[baseline]['bad_100cp_moves']
        assert summary['trained25']['allows_mate']<=summary[baseline]['allows_mate']
    if (ART/'vs-heuristic').exists():raise FileExistsError('Preserve games')
    write(ART/'exploratory-pilot-amendment.json',{'reason':'Positive validation ranking gain withzerooldNNregressions; equal-depth candidate matched oldNN choices, but was slightly worse thanheuristic, so initialstrictimprovementgatefailed. Run20exploratoryfreshgames to measure250ms behavior without claiming screenimprovement. Amendment recorded afterscreen/beforeanygameoutcomes.','extension':'Unchanged>=60%/20complete/errorfree gate triggers80newconfirmationgames. Report exploratory20, independent80 andcombined100 separately. OldNNsecondary onlyifpilotpassed. Noautomaticpromotion.'})
    write(ART/'frozen-selection.json',{'checkpoint_sha256':digest(CP),'weight':.25,'quiet_only':True,'incremental':True,'fast_features':True,'exploratory_after_tied_depth_screen':True})
    def run(args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    with exclusive_cpu('v22 exploratory equal-time strength'):
        opening='benchmarks/openings-search-smooth-v22-pilot.json'
        run(['-m','benchmarks.generate_strength_openings','--output',opening,'--pairs','10','--seed','222500','--exclude-data',str(DATA)])
        common=['-m','ml.compare','--checkpoint',str(CP),'--nn-weight','.25','--quiet-only','--incremental','--fast-features','--opponent-checkpoint',str(OLD),'--opponent-quiet-only','--opponent-frozen-inference','--time-ms','250','--max-plies','1000']
        out=ART/'vs-heuristic';write(ART/'status.json',{'phase':'exploratory-heuristic-pilot','strict_screen_qualified':screen['qualified_for_games']})
        run(common+['--opponent-nn-weight','0','--opponent-incremental','--openings',opening,'--pairs','10','--output',str(out)])
        report=json.loads((out/'report.json').read_text());matches={'pilot':report['summary']};extend=matches['pilot']['completed']==20 and not matches['pilot']['errors'] and matches['pilot']['score_fraction_completed']>=.6
        if extend:
            opening2='benchmarks/openings-search-smooth-v22-extension.json';run(['-m','benchmarks.generate_strength_openings','--output',opening2,'--pairs','40','--seed','222501','--exclude-data',str(DATA)])
            out2=ART/'vs-heuristic-extension';write(ART/'status.json',{'phase':'100-game-confirmation'})
            run(common+['--opponent-nn-weight','0','--opponent-incremental','--openings',opening2,'--pairs','40','--output',str(out2)])
            extra=json.loads((out2/'report.json').read_text());matches['extension']=extra['summary']
            for g in extra['games']:g['pair']+=10
            matches['combined100']=summarize(report['games']+extra['games'])
            out3=ART/'vs-old-NN';run(common+['--opponent-nn-weight','.25','--openings',opening,'--pairs','10','--output',str(out3)]);matches['oldNN']=json.loads((out3/'report.json').read_text())['summary']
        write(ART/'status.json',{'phase':'completed','strict_screen_qualified':screen['qualified_for_games'],'exploratory_pilot':True,'matches':matches,'extended_to_100':extend,'app_promoted':False})
if __name__=='__main__':main()
