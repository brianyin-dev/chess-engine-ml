"""Initialize a cheaper board-feature network from existing evaluator weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION


def project(saved, hidden_sizes=(32,16)):
    if saved.get('target_mode') != 'residual' or saved.get('diagnostic_only'):
        raise ValueError('projection requires a nondiagnostic residual evaluator')
    state=saved['state_dict']
    first_width,second_width=hidden_sizes
    if not (0<first_width<=state['net.0.weight'].shape[0] and
            0<second_width<=state['net.2.weight'].shape[0] and saved['input_size']>=INPUT_SIZE):
        raise ValueError('projection must shrink compatible widths and preserve board features')
    first=torch.argsort(state['net.2.weight'].abs().sum(0),descending=True)[:first_width]
    second=torch.argsort(state['net.4.weight'][0].abs(),descending=True)[:second_width]
    model=ChessNet(INPUT_SIZE,saved.get('correction_limit_cp'),saved.get('color_consistent',False),hidden_sizes)
    projected=model.state_dict()
    projected['net.0.weight']=state['net.0.weight'][first,:INPUT_SIZE]
    projected['net.0.bias']=state['net.0.bias'][first]
    projected['net.2.weight']=state['net.2.weight'][second][:,first]
    projected['net.2.bias']=state['net.2.bias'][second]
    projected['net.4.weight']=state['net.4.weight'][:,second]
    projected['net.4.bias']=state['net.4.bias']
    result={**saved,'version':MODEL_VERSION,'input_size':INPUT_SIZE,'hidden_sizes':list(hidden_sizes),'state_dict':projected}
    return result,{'first_units':first.tolist(),'second_units':second.tolist(),
                   'policy':'Keep highest downstream absolute-weight units; remove relationship input columns. No held-out example selection.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hidden-sizes',type=int,nargs=2,default=(32,16))
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.projection.json').exists():p.error('output must be new')
    result,report=project(torch.load(args.source,map_location='cpu',weights_only=True),tuple(args.hidden_sizes))
    report['source_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
    args.output.with_suffix('.projection.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
