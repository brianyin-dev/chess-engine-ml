"""Audit new family isolation, explicit Torch parity, and terminal game replay."""
import json
import chess,chess.pgn,torch
from ml.run_generalization_v21 import ART,NEW,DATA,OLD,CP,ROOT,previous,readrows,write
from ml.audit_leaf_speed_v20 import aliases
from ml.dataset import correction_factor
from ml.model import board_to_tensor,SCORE_SCALE
from ml.evaluator import NeuralEvaluator
from engine.evaluation import evaluate_position

def main():
    sets={s:aliases(DATA,s) for s in ('train','val','test')}
    overlap={f'{a}-{b}':{'positions':len(sets[a][0]&sets[b][0]),'families':len(sets[a][1]&sets[b][1])} for a,b in [('train','val'),('train','test'),('val','test')]}
    parity={};fens={r['fen'] for s in ('val','test') for r in readrows(NEW/f'{s}.jsonl')}
    for name,path in [('old',OLD),('trained',CP),('trained25',ART/'trained25.pt')]:
        if not path.exists():continue
        weight=.25 if name=='trained25' else .05
        model=NeuralEvaluator(path,weight,True,incremental=True,fast_features=True);errors=[]
        with torch.inference_mode():
            for fen in fens:
                b=chess.Board(fen);score=round(evaluate_position(b)+model.model(board_to_tensor(b,model.input_size)).item()*SCORE_SCALE*correction_factor(b,weight,True))
                if model.evaluate_position(b)!=score:errors.append(fen)
        parity[name]={'positions':len(fens),'mismatches':errors}
    games={}
    for folder in ART.glob('vs-*'):
        p=folder/'games.pgn'
        if not p.exists():continue
        count=0
        with p.open() as handle:
            while (game:=chess.pgn.read_game(handle)) is not None:
                assert not game.errors;b=game.board()
                for move in game.mainline_moves():assert move in b.legal_moves;b.push(move)
                assert b.result(claim_draw=True)==game.headers['Result'];count+=1
        games[folder.name]=count
    original=NeuralEvaluator(OLD).model.state_dict()
    unchanged={}
    for name,path in [('selected5',CP),('selected25',ART/'trained25.pt')]:
        if path.exists():
            state=NeuralEvaluator(path).model.state_dict()
            unchanged[name]=all(torch.equal(original[k],state[k]) for k in original)
    frozen=json.loads((ART/'frozen-selection.json').read_text())
    from ml.run_balanced_v19 import digest
    source_changes=[p for p,h in frozen['source_sha256'].items() if digest(ROOT/p)!=h]
    result={'unchanged_selected_weights':unchanged,'fixed_inference_source_changes':source_changes,'cross_split_overlap':overlap,'explicit_torch_parity':parity,'legal_terminal_games':games}
    write(ART/'final-audit.json',result);print(json.dumps(result,indent=2))
    assert not source_changes
    assert not any(any(v.values()) for v in overlap.values())
    assert not any(v['mismatches'] for v in parity.values())
if __name__=='__main__':main()
