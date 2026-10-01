"""Audit adaptation isolation, exact Torch inference, and source stability."""
import json
import chess,chess.pgn,torch
from ml.run_search_v22 import ROOT,ART,DATA,NEW,OLD,CP,labels
from ml.run_balanced_v19 import readrows,write,digest
from ml.audit_leaf_speed_v20 import aliases
from ml.evaluator import NeuralEvaluator
from ml.model import board_to_tensor,SCORE_SCALE
from ml.dataset import correction_factor
from engine.evaluation import evaluate_position,evaluate
from ml.evaluate_blends import pair_metrics

def main():
    sets={s:aliases(DATA,s) for s in ('train','val','test')}
    overlap={f'{a}-{b}':{'positions':len(sets[a][0]&sets[b][0]),'source_ids':len(sets[a][1]&sets[b][1])} for a,b in [('train','val'),('train','test'),('val','test')]}
    newsets={s:aliases(NEW,s) for s in sets}
    prior=set()
    for p in (ROOT/'ml/data').glob('**/*.jsonl'):
        if p.is_relative_to(NEW) or p.is_relative_to(DATA):continue
        for r in readrows(p):prior.update(labels.canonical(r[f]) for f in ('fen','root_fen','good_fen','bad_fen') if f in r)
    prior_aliases={s:len(newsets[s][0]&prior) for s in sets}
    fens={r['fen'] for s in ('val','test') for r in readrows(NEW/f'{s}.jsonl')}
    parity={}
    for name,path in [('old25',OLD),('trained25',CP)]:
        m=NeuralEvaluator(path,.25,True,incremental=True,fast_features=True);errors=[]
        with torch.inference_mode():
            for fen in fens:
                b=chess.Board(fen);score=round(evaluate_position(b)+m.model(board_to_tensor(b,m.input_size)).item()*SCORE_SCALE*correction_factor(b,.25,True))
                if score!=m.evaluate_position(b):errors.append(fen)
        parity[name]={'positions':len(fens),'mismatches':errors}
    original=NeuralEvaluator(OLD).model.state_dict();trained=NeuralEvaluator(CP).model.state_dict()
    changed=any(not torch.equal(original[k],trained[k]) for k in original)
    hashes=json.loads((ART/'fixed-source-hashes.json').read_text());changes=[p for p,h in hashes.items() if digest(ROOT/p)!=h]
    cases=json.loads((ART/'frozen-move-screen.json').read_text())
    screen_train_aliases=sum(labels.canonical(r['fen']) in sets['train'][0] for r in cases)
    games={}
    for folder in ART.glob('vs-*'):
        p=folder/'games.pgn'
        if not p.exists():continue
        count=0
        with p.open() as h:
            while (g:=chess.pgn.read_game(h)) is not None:
                assert not g.errors;b=g.board()
                for move in g.mainline_moves():assert move in b.legal_moves;b.push(move)
                assert b.result(claim_draw=True)==g.headers['Result'];count+=1
        games[folder.name]=count
    report={'canonical_and_source_id_cross_split_overlap':overlap,'new_prior_label_aliases':prior_aliases,'explicit_torch_integer_parity':parity,'learned_weights_changed':changed,'fixed_source_changes':changes,'move_screen_training_aliases':screen_train_aliases,'legal_terminal_games':games,'scope':'New source opening families isolated; canonical labels/prior aliases checked. Legacy family provenance incomplete; this is not proof of globally unseen opening families.'}
    models={'heuristic':evaluate,'old25':NeuralEvaluator(OLD,.25,True,incremental=True,fast_features=True),'trained25':NeuralEvaluator(CP,.25,True,incremental=True,fast_features=True)}
    write(ART/'position-metrics.json',{n:{s:pair_metrics(readrows(NEW/'pairs'/f'{s}.jsonl'),m) for s in ('train','val','test')} for n,m in models.items()})
    write(ART/'final-audit.json',report);print(json.dumps(report,indent=2))
    assert not any(any(v.values()) for v in overlap.values()) and not any(prior_aliases.values())
    assert changed and not changes and not screen_train_aliases
    assert not any(v['mismatches'] for v in parity.values())
if __name__=='__main__':main()
