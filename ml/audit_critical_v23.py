"""Reproducible leakage, runtime parity and frozen-source checks for v23."""
import hashlib,json
import chess,torch
from ml.train_critical_v23 import ART,DATA,CP,ROOT,board,baseline_models
from ml.run_balanced_v19 import readrows,write
from ml.run_search_v22 import labels
from ml.evaluator import NeuralEvaluator
from ml.model import board_to_tensor,SCORE_SCALE
from ml.dataset import correction_factor
from engine.evaluation import evaluate_position
from engine.search import _Search,INF
from ml.diagnose_critical_v23 import forced
from benchmarks.cpu_lock import exclusive_cpu

def aliases(rows):
    return {labels.canonical(r[f]) for r in rows for f in ('fen','root_fen','good_fen','bad_fen') if f in r}

def main():
    with exclusive_cpu('v23 final reproducibility audit'):
        train=readrows(DATA/'train.jsonl')+readrows(DATA/'pairs/train.jsonl')
        val=readrows(DATA/'val.jsonl')+readrows(DATA/'pairs/val.jsonl')
        vr=json.loads((ART/'validation-roots.json').read_text())
        tr=json.loads((ART/'test-roots.json').read_text())
        sets={'train':aliases(train),'validation':aliases(val),'validation_roots':aliases(vr),'test_roots':aliases(tr)}
        overlaps={f'{a}:{b}':len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','validation_roots'),('train','test_roots'),('validation_roots','test_roots')]}
        assert not any(overlaps.values()),overlaps
        families=set(json.loads((DATA/'manifest.json').read_text())['target_families'])
        assert not families & {r['game_id'] for r in val+vr+tr}
        hashes=json.loads((ROOT/'ml/artifacts/search-smooth-v22/fixed-source-hashes.json').read_text())
        changed=[p for p,h in hashes.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        assert not changed,changed
        m=NeuralEvaluator(CP,.25,True,incremental=True,fast_features=True)
        fens={r['fen'] for r in train+val+vr+tr if 'fen' in r}
        mismatches=[]
        with torch.inference_mode():
            for fen in fens:
                b=chess.Board(fen)
                prediction=m.model(board_to_tensor(b,m.input_size).unsqueeze(0)).item()*SCORE_SCALE
                expected=round(evaluate_position(b)+prediction*correction_factor(b,.25,True))
                if m.evaluate_position(b)!=expected:mismatches.append(fen)
        assert not mismatches,mismatches[:3]
        traces=0
        model=baseline_models()['heuristic']
        for r in json.loads((ART/'fatal-and-reachable.json').read_text())['games']:
            f=r['failure']
            if not f:continue
            for move in {f['played'],f['review']['best_move']}:
                b=board(f);history=list(b.move_stack);fen=b.fen();mv=chess.Move.from_uci(move)
                traced=forced(b,mv,model,f['matched_depth'])
                worker=_Search(b,model,None,False)
                with worker.pushed(b,mv):expected=-worker.negamax(b,f['matched_depth']-1,-INF,INF,1)
                assert traced['root_score_cp']==expected,(r['source_game'],move,traced['root_score_cp'],expected)
                assert b.fen()==fen and b.move_stack==history
                traces+=1
        write(ART/'audit.json',{'canonical_alias_overlaps':overlaps,'target_families_excluded_from_holdouts':True,'fixed_runtime_sources_changed':changed,'torch_runtime_positions':len(fens),'torch_runtime_mismatches':len(mismatches),'full_window_forced_trace_comparisons':traces})
        print(json.dumps(json.loads((ART/'audit.json').read_text()),indent=2))

if __name__=='__main__':main()
