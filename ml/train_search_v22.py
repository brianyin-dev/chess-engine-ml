"""Smooth teacher supervision with exact rounded validation and old-NN protection."""
import copy,json
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ml.dataset import ChessEvalDataset
from ml.model import SCORE_SCALE
from ml.evaluator import NeuralEvaluator
from ml.train import pair_loader,rounded_score
from ml.run_balanced_v19 import readrows,write,digest


def smooth_pair_loss(gap_cp,teacher_gap_cp):
    # Monotone expected-score proxy; not calibrated Stockfish WDL probabilities.
    return nn.functional.binary_cross_entropy_with_logits(gap_cp/100.,torch.sigmoid(teacher_gap_cp.clamp(max=500)/100.),reduction='none')


def protection_loss(gap_cp,reference_gap_cp):
    # Protect rankings the original NN gets right, without freezing its margins.
    return torch.where(reference_gap_cp>0,torch.relu(reference_gap_cp.clamp(max=10)-gap_cp)/100.,torch.zeros_like(gap_cp))


def train(data,checkpoint,metrics,old):
    torch.set_num_threads(1);torch.manual_seed(42)
    reference=NeuralEvaluator(old,.25,True).model
    model=copy.deepcopy(reference).train();reference.eval().requires_grad_(False)
    datasets={s:ChessEvalDataset(readrows(data/f'{s}.jsonl'),'residual',model.net[0].in_features,.25,True) for s in ('train','val','test')}
    loaders={s:pair_loader(data/'pairs'/f'{s}.jsonl',s=='train',model.net[0].in_features,'residual',.25,True,True) for s in datasets}
    teacher={s:torch.tensor([r['cp_loss'] for r in readrows(data/'pairs'/f'{s}.jsonl')]) for s in datasets}
    # Build a paired dataset containing teacher magnitudes; preserve row order.
    paired=loaders['train'].dataset.tensors
    loader=DataLoader(TensorDataset(*paired,teacher['train']),batch_size=64,shuffle=True)
    ds=datasets['train'];scoreloader=DataLoader(TensorDataset(ds.features,ds.targets,ds.output_factors,ds.target_baselines),batch_size=64,shuffle=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00005)
    history=[];bestkey=None;beststate=None;bestepoch=None;stale=0
    def validation(s):
        m=model.eval();correct=regressions=total=0;brier=0.
        with torch.inference_mode():
            for batch in loaders[s]:
                good,bad,_,sign,gf,bf,gb,bb,_=batch
                gap=sign*(rounded_score(gb,gf*m(good))-rounded_score(bb,bf*m(bad)))
                ref=sign*(rounded_score(gb,gf*reference(good))-rounded_score(bb,bf*reference(bad)))
                correct+=(gap>0).sum().item();regressions+=((ref>0)&(gap<=0)).sum().item();total+=len(gap)
            d=datasets[s]
            pred=d.target_baselines+d.output_factors*m(d.features)*SCORE_SCALE
            target=d.target_baselines+d.targets*SCORE_SCALE
            brier=nn.functional.mse_loss(torch.sigmoid(pred/400),torch.sigmoid(target/400)).item()
        return {'pairs':total,'correct':correct,'regressions_from_old':regressions,'teacher_probability_proxy_mse':brier,'selection_key':[-(correct-2*regressions),-correct,brier]}
    initial=validation('val');write(metrics.with_name('initial-validation.json'),initial)
    for epoch in range(1,31):
        model.train()
        for x,y,f,base in scoreloader:
            optimizer.zero_grad();pred=base+f*model(x)*SCORE_SCALE;target=base+y*SCORE_SCALE
            probability_loss=nn.functional.mse_loss(torch.sigmoid(pred/400.),torch.sigmoid(target/400.))
            with torch.no_grad():oldpred=f*reference(x)
            anchor=nn.functional.mse_loss(f*model(x),oldpred)
            (4*probability_loss+2*anchor).backward();optimizer.step()
        for good,bad,_,sign,gf,bf,gb,bb,_,cp in loader:
            optimizer.zero_grad();gap=sign*(gb+gf*model(good)*SCORE_SCALE-bb-bf*model(bad)*SCORE_SCALE)
            with torch.no_grad():refgap=sign*(gb+gf*reference(good)*SCORE_SCALE-bb-bf*reference(bad)*SCORE_SCALE)
            loss=(smooth_pair_loss(gap,cp)+8*protection_loss(gap,refgap)).mean()
            loss.backward();optimizer.step()
        val=validation('val');history.append({'epoch':epoch,**val});print('Smooth epoch',epoch,val,flush=True)
        k=tuple(val['selection_key'])
        if bestkey is None or k<bestkey:
            bestkey=k;beststate={n:t.detach().clone() for n,t in model.state_dict().items()};bestepoch=epoch;stale=0
        else:
            stale+=1
            if stale>=8:break
    model.load_state_dict(beststate)
    saved=torch.load(old,map_location='cpu',weights_only=True)
    saved.update(state_dict=beststate,training_correction_weight=.25,training_quiet_only=True)
    torch.save(saved,checkpoint)
    write(metrics,{'initial_checkpoint_sha256':digest(old),'data_manifest_sha256':digest(data/'manifest.json'),'best_epoch':bestepoch,'initial_validation':initial,'selected_validation':validation('val'),'test_descriptive_only':validation('test'),'history':history,'selection':'Best UPDATED epoch by rounded validation correct minus2 oldNN regressions, then correct count, then probability-proxy MSE. Original retained as unchanged baseline, never silently substituted. Candidate need not qualify for games.','objective':'Smooth sigmoid teacher score proxy(400cp), teacher gap logistic(100cp), old-NN correct-ranking protection8, anchor2. Proxy is not calibrated WDL. Runtime quiet25% blend retained; no rounding in loss, exact whole-endpoint rounding in validation.'})
