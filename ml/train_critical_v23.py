"""Focused feasible-oracle distillation, retention, searched validation and holdout."""
import copy,json,random
from collections import defaultdict
import chess,chess.engine,torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from engine.evaluation import evaluate_position,evaluate
from engine.search import search
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.analyze import score_data
from ml.diagnose_critical_v23 import ART,SF,OLD,PREVIOUS,forced,review
from ml.confirm_fatal_v23 import StrongBoundedTeacher,fatal
from ml.run_search_v22 import ROOT,exact,labels
from ml.run_balanced_v19 import readrows,write,digest
from ml.evaluator import NeuralEvaluator
from ml.model import board_to_tensor,SCORE_SCALE
from ml.dataset import ChessEvalDataset,correction_factor
from ml.train_search_v22 import smooth_pair_loss,protection_loss
from ml.train import pair_loader,rounded_score
NEW=ROOT/'ml/data/critical-v23-new';DATA=ROOT/'ml/data/critical-v23-2026';CP=ART/'trained.pt'


def board(record):
    b=chess.Board(record['initial_fen'])
    for u in record['history']:b.push_uci(u)
    return b


def baseline_models():
    return {'heuristic':NeuralEvaluator(OLD,0,True,incremental=True,fast_features=True),'old25':NeuralEvaluator(OLD,.25,True,incremental=True,fast_features=True),'v22':NeuralEvaluator(PREVIOUS,.25,True,incremental=True,fast_features=True)}


def choices(sf,cases,models,depth3=True):
    records=[]
    for index,r in enumerate(cases):
        b=board(r);depth=3 if depth3 else r['matched_depth'];cache={};probes={}
        for name,m in models.items():
            p=search(b.copy(stack=True),depth=depth,eval_fn=m)
            assert p.depth==depth or abs(p.score or 0)>28000
            if p.move.uci() not in cache:cache[p.move.uci()]=review(sf,b,p.move,256000)
            probes[name]={'move':p.move.uci(),'depth':p.depth,'review':cache[p.move.uci()]}
        records.append({'root':r,'probes':probes})
    comparable=[r for r in records if all(p['review']['cp_loss'] is not None for p in r['probes'].values())]
    summary={n:{'positions':len(records),'comparable':len(comparable),'mean_cp_regret':sum(r['probes'][n]['review']['cp_loss'] for r in comparable)/len(comparable) if comparable else None,'losing_transitions':sum(fatal(r['probes'][n]['review']) for r in records),'bad_150cp_moves':sum((r['probes'][n]['review']['cp_loss'] or 0)>=150 for r in records)} for n in models}
    return {'records':records,'summary':summary}


def prepare(sf):
    rng=random.Random(230023);games=json.loads((ART/'fatal-and-reachable.json').read_text())['games']
    targets=[r for r in games if r['failure'] and r['failure'].get('oracle',{}).get('demonstrated_repair')]
    if not targets:raise ValueError('No demonstrated bounded repairs; do not train irrelevant positions')
    trainfamilies={r['game_id'] for r in targets};prior=set()
    for path in (ROOT/'ml/data').glob('**/*.jsonl'):
        for r in readrows(path):prior.update(labels.canonical(r[f]) for f in ('fen','root_fen','good_fen','bad_fen') if f in r)
    rows={};pairs=[];lessons=[]
    for r in targets:
        f=r['failure'];oracle=StrongBoundedTeacher(sf);oracle.rows=f['oracle']['cache'].copy()
        # Trace the successful corrected action and originalNN action atactualdepth.
        good=forced(board(f),chess.Move.from_uci(f['oracle']['move']),oracle,f['matched_depth'])
        oldmodel=baseline_models()['v22'];bad=forced(board(f),chess.Move.from_uci(f['probes']['new25']['move']),oldmodel,f['matched_depth'])
        candidates=list(oracle.rows);rng.shuffle(candidates)
        special=[x['leaf']['fen'] for x in (good,bad) if x['leaf']]
        candidates=list(dict.fromkeys(special+candidates))
        for fen in candidates:
            b=chess.Board(fen);k=labels.canonical(fen)
            if k in prior or k in rows or b.is_game_over() or not correction_factor(b,.25,True):continue
            # Retrieve64klabels withfulloriginal history ifavailable inPV; for
            # othercalls raw cachedteacher provenance is retained.
            if fen not in oracle.rows:
                source=next(x['leaf'] for x in (good,bad) if x['leaf'] and x['leaf']['fen']==fen)
                b=chess.Board(f['initial_fen'])
                for u in source['history']:b.push_uci(u)
                oracle.evaluate_position(b)
            teacher=oracle.rows[fen];base=evaluate_position(chess.Board(fen));limit=62.5
            delta=(limit if teacher['mate']>0 else -limit) if teacher['score_cp'] is None else max(-limit,min(limit,teacher['score_cp']-base))
            rows[k]={'fen':fen,'score_cp':round(base+delta),'teacher_score_cp':teacher['score_cp'],'teacher_mate':teacher['mate'],'game_id':r['game_id'],'ply':b.ply(),'source':'confirmed_bounded_rook_endgame_repair','training_weight':8,'root_fen':f['fen']}
            if sum(x['game_id']==r['game_id'] for x in rows.values())>=128:break
        if good['leaf'] and bad['leaf']:
            ga,ba=good['leaf']['fen'],bad['leaf']['fen'];gk,bk=labels.canonical(ga),labels.canonical(ba)
            if gk in rows and bk in rows:
                sign=1 if board(f).turn else -1;gap=sign*(rows[gk]['score_cp']-rows[bk]['score_cp'])
                # Target anactualsearched backup margin within thefeasible hybrid.
                if gap>0:pairs.append({'good_fen':ga,'bad_fen':ba,'root_fen':f['fen'],'sign':sign,'game_id':r['game_id'],'cp_loss':gap,'heuristic_gap_cp':sign*(evaluate(chess.Board(ga))-evaluate(chess.Board(ba))),'source':'searched_bounded_repair','training_weight':32})
        lessons.append({'source_game':r['source_game'],'game_id':r['game_id'],'root':f,'good_trace':good,'bad_trace':bad,'new_rows':sum(x['game_id']==r['game_id'] for x in rows.values())})
    if len(rows)<16:raise ValueError('Insufficient novel repairlabels')
    # Whole targetfamilies become development/train, includingformerpilotval.
    # Their familyids are excluded fromall newvalidation andtestselection.
    roots=json.loads((ROOT/'ml/artifacts/search-smooth-v22/roots.json').read_text());rng.shuffle(roots)
    validation=[];seen=set();counts=defaultdict(int)
    for r in roots:
        k=labels.canonical(r['fen'])
        if r['split']!='val' or r['game_id'] in trainfamilies or k in prior or k in rows or k in seen or counts[r['game_id']]>=2:continue
        if board(r).is_game_over():continue
        seen.add(k);counts[r['game_id']]+=1;validation.append(r)
        if len(validation)==16:break
    if len(validation)<12:raise ValueError('Insufficient novel family-separated validationroots')
    test=json.loads((ROOT/'ml/artifacts/search-smooth-v22/frozen-move-screen.json').read_text())
    test=[r for r in test if r['game_id'] not in trainfamilies and labels.canonical(r['fen']) not in rows and labels.canonical(r['fen']) not in seen]
    # Addpilot testfatal position asaseparate targeted challenge, nevertraining.
    for r in games:
        if r['split']=='test' and r['failure'] and r['game_id'] not in trainfamilies:
            f=r['failure'];test.append({**f,'game_id':r['game_id'],'split':'test','targeted_fatal_holdout':True})
    forbidden={labels.canonical(r['fen']) for r in validation+test}
    assert not (set(rows)&forbidden)
    base=ROOT/'ml/data/search-smooth-v22-2026'
    retention_scores=[r for r in readrows(base/'train.jsonl') if labels.canonical(r['fen']) not in forbidden]
    retention_pairs=[r for r in readrows(base/'pairs/train.jsonl') if not any(labels.canonical(r[f]) in forbidden for f in ('root_fen','good_fen','bad_fen') if f in r)]
    rng.shuffle(retention_scores);rng.shuffle(retention_pairs);retention_scores=retention_scores[:512];retention_pairs=retention_pairs[:512]
    # Equalattention scoretargets for retention deliberately preserve v22 scores,
    # rather than re-solvinglegacy labels duringthisdiagnostic correction run.
    ref=baseline_models()['v22']
    for r in retention_scores:r.update(score_cp=ref(chess.Board(r['fen'])),source='v22_prediction_retention',training_weight=1)
    validation_rows=[r for r in readrows(base/'val.jsonl') if r['game_id'] not in trainfamilies and labels.canonical(r['fen']) not in rows]
    validation_pairs=[r for r in readrows(base/'pairs/val.jsonl') if r['game_id'] not in trainfamilies and not any(labels.canonical(r[f]) in rows for f in ('root_fen','good_fen','bad_fen') if f in r)]
    NEW.mkdir();DATA.mkdir();(DATA/'pairs').mkdir()
    (NEW/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows.values()))
    (NEW/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    for s,scores,ps in [('train',retention_scores+list(rows.values()),retention_pairs+pairs),('val',validation_rows,validation_pairs)]:
        (DATA/f'{s}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores));(DATA/'pairs'/f'{s}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ps))
    write(ART/'training-lessons.json',lessons);write(ART/'validation-roots.json',validation);write(ART/'test-roots.json',test)
    manifest={'target_games':len(targets),'target_families':sorted(trainfamilies),'new_rows':len(rows),'new_pairs':len(pairs),'retention_rows':len(retention_scores),'retention_pairs':len(retention_pairs),'validation_roots':len(validation),'test_roots':len(test),'policy':'All demonstrated repairpilot cases aredevelopment/train (includingformerpilotval). Bothfamilies removed fromnewvalidation/test. Newteacher targets areFEASIBLE64k boundedoracle scores, withrawSFcp/mate preserved; no claim ofSFscorecalibration.512v22prediction-retention rows and512legacyteacher pairs. WeightedCFrows8,pairs32. Canonicalnewlabels excluded fromprior; rootheldouts excluded fromretention. Currentfamilies isolated, legacyfamily provenance incomplete.'}
    write(DATA/'manifest.json',manifest);write(NEW/'manifest.json',manifest)


def train(sf):
    torch.set_num_threads(1);torch.manual_seed(42)
    reference=NeuralEvaluator(PREVIOUS,.25,True).model;model=copy.deepcopy(reference).train();reference.eval().requires_grad_(False)
    rows=readrows(DATA/'train.jsonl');ds=ChessEvalDataset(rows,'residual',reference.net[0].in_features,.25,True)
    weights=torch.tensor([r.get('training_weight',1) for r in rows],dtype=torch.float32)
    loader=DataLoader(TensorDataset(ds.features,ds.targets,ds.output_factors,ds.target_baselines),batch_size=64,sampler=WeightedRandomSampler(weights,len(rows),replacement=True))
    pair=pair_loader(DATA/'pairs/train.jsonl',True,reference.net[0].in_features,'residual',.25,True,True)
    valpair=pair_loader(DATA/'pairs/val.jsonl',False,reference.net[0].in_features,'residual',.25,True,True)
    teacher=torch.tensor([r['cp_loss'] for r in readrows(DATA/'pairs/train.jsonl')])
    pl=DataLoader(TensorDataset(*pair.dataset.tensors,teacher),batch_size=64,shuffle=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
    validation=json.loads((ART/'validation-roots.json').read_text());baselines=choices(sf,validation,baseline_models());write(ART/'validation-baselines.json',baselines)
    lessons=json.loads((ART/'training-lessons.json').read_text());history=[];best=None;beststate=None;bestepoch=None
    # Save every10epochs. Selection usesUNSEENvalidation SEARCH choices, then
    # retentionpair regressions, then training repairchoices asatie-breaker.
    def save(path):
        saved=torch.load(PREVIOUS,map_location='cpu',weights_only=True);saved.update(state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},training_correction_weight=.25,training_quiet_only=True);torch.save(saved,path)
    for epoch in range(1,61):
        model.train()
        for x,y,f,base in loader:
            optimizer.zero_grad();prediction=f*model(x)
            with torch.no_grad():old=f*reference(x)
            # Newtargets alreadyfeasible; MSE trains the boundedcorrection itself.
            loss=16*(prediction-y).square().mean()+.5*(prediction-old).square().mean()
            loss.backward();optimizer.step()
        for good,bad,_,sign,gf,bf,gb,bb,importance,cp in pl:
            optimizer.zero_grad();gap=sign*(gb+gf*model(good)*SCORE_SCALE-bb-bf*model(bad)*SCORE_SCALE)
            with torch.no_grad():refgap=sign*(rounded_score(gb,gf*reference(good))-rounded_score(bb,bf*reference(bad)))
            loss=((smooth_pair_loss(gap,cp)+4*protection_loss(gap,refgap))*importance).sum()/importance.sum()
            loss.backward();optimizer.step()
        if epoch%10:continue
        path=ART/f'epoch-{epoch}.pt';save(path);adapter=NeuralEvaluator(path,.25,True,incremental=True,fast_features=True)
        v=choices(sf,validation,{'candidate':adapter});summary=v['summary']['candidate']
        repairs=[]
        for lesson in lessons:
            root=lesson['root'];p=search(board(root),depth=root['matched_depth'],eval_fn=adapter);r=review(sf,board(root),p.move,256000)
            repairs.append({'source_game':lesson['source_game'],'move':p.move.uci(),'review':r,'avoids_losing_flag':not fatal(r)})
        regressions=0
        model.eval()
        with torch.inference_mode():
            for good,bad,_,sign,gf,bf,gb,bb,_ in valpair:
                g=sign*(rounded_score(gb,gf*model(good))-rounded_score(bb,bf*model(bad)));old=sign*(rounded_score(gb,gf*reference(good))-rounded_score(bb,bf*reference(bad)))
                regressions+=((old>0)&(g<=0)).sum().item()
        key=(summary['losing_transitions'],summary['mean_cp_regret'],regressions,-sum(r['avoids_losing_flag'] for r in repairs))
        record={'epoch':epoch,'validation':v,'retention_regressions':regressions,'training_repairs':repairs,'selection_key':key};history.append(record);write(ART/'training-progress.json',history);print('Critical epoch',epoch,'val',summary,'retentionreg',regressions,'trainfixed',sum(r['avoids_losing_flag'] for r in repairs),flush=True)
        if best is None or key<best:best=key;beststate={k:v.detach().clone() for k,v in model.state_dict().items()};bestepoch=epoch
    model.load_state_dict(beststate);save(CP);write(ART/'training.json',{'best_epoch':bestepoch,'history':history,'selection':'Unseen validation searched fatalflags/regret, retention-regressions, trainingrepairs tie-break.6fixedepochs10..60, no testselection. Initialv22 remainsbaseline, no silentreplacement.','architecture':'Unchanged892/64/32,quiet25%,bounded250cp. Fixedv20inference.','loss':'Feasibleoracle correctionMSE16+anchor.5, weightedCFrows8; smoothranking+roundedreference protection4 withCFpairweights32; lr.0002,seed42.'})


def test(sf):
    candidates=baseline_models();candidates['candidate']=NeuralEvaluator(CP,.25,True,incremental=True,fast_features=True)
    cases=json.loads((ART/'test-roots.json').read_text());result=choices(sf,cases,candidates);write(ART/'unseen-move-test.json',result)
    s=result['summary'];a=s['candidate'];qualified=a['comparable']>=24 and a['mean_cp_regret']<s['v22']['mean_cp_regret'] and a['mean_cp_regret']<=s['heuristic']['mean_cp_regret'] and a['losing_transitions']<=s['heuristic']['losing_transitions'] and a['bad_150cp_moves']<=s['heuristic']['bad_150cp_moves']
    write(ART/'status.json',{'phase':'completed','qualified_for_games':qualified,'unseen_move_summary':s,'app_promoted':False});return qualified


def main():
    if DATA.exists() or CP.exists():raise FileExistsError('Preserve focusedexperiment')
    protocol={'pilot_is_development':'Use bothdemonstratedrook repairsfortraining, includingformerpilotvalcase. Exclude their wholefamilies fromnewvalidation/test. Originalpilot isnot afinalholdout.','data':'Max128newquiet64kboundedteacher targets/repair pluscriticalPVleaves;512priorpredictionretention rows/512priorrankpairs; caseweights8/32. Priorcanonicalaliases excluded; heldoutrootaliases removed fromretention.','selection':'Freeze16novel validationroots otherfamilies,32priornoveltestroots plus1pilot-testfatalroot. Pickamong6fixedepochs usingvalidationsearched choices, thenretentionregressions, then trainingrootrepairs. Finalunseen testonlyonce.','gate':'Require lowermean256knodeteacher regret thanunchangedv22 andnotworseheuristic, noextra losingtransitions/150cpblunders before ANY newgamepilot. Noadaptive pilotbypass.'}
    write(ART/'focused-training-protocol.json',protocol)
    with exclusive_cpu('v23 focusedrepair training and unseen searcheddecision gate'):
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32});prepare(sf);train(sf);return test(sf)
if __name__=='__main__':main()
