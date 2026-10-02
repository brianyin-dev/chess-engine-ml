"""Explain the bishop trace and screen a general advanced-queen confinement term."""
import json
from pathlib import Path
import chess
from benchmarks import evaluation_baseline_v25 as base

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'benchmarks/results/bishop-v25'

def queen_confinement(board,color,penalty=220):
    """Penalize advanced queens with at most one unattacked destination.

    A conservative geometric risk feature, not proof of a trapped queen.
    Home/second-rank queens are exempt to avoid penalizing normal development.
    """
    queens=board.pieces_mask(chess.QUEEN,color)
    advanced=[sq for sq in chess.scan_forward(queens)
              if (chess.square_rank(sq) if color else 7-chess.square_rank(sq))>=2]
    if not advanced:return 0
    attacks=0
    for sq in chess.scan_forward(board.occupied_co[not color]):attacks|=board.attacks_mask(sq)
    score=0
    for sq in advanced:
        safe=board.attacks_mask(sq)&~board.occupied_co[color]&~attacks&~board.kings
        if safe.bit_count()<=1:score-=penalty
    return score

class Evaluator:
    cacheable_by_fen=True
    def __init__(self,penalty=0):self.penalty=penalty
    def evaluate_position(self,b):
        if not self.penalty:return base.evaluate_position(b)
        return base.evaluate_position(b)+queen_confinement(b,True,self.penalty)-queen_confinement(b,False,self.penalty)
    __call__=evaluate_position

def components(b):
    phase=base._phase(b)
    functions={'material':base._material_score,'mobility':base._mobility_score,
        'piece_square':lambda b,c:base._pst_score(b,c,phase),
        'development':lambda b,c:base._development_score(b,c,phase),
        'center':base._center_control_score,'rook_files':base._rook_activity_score,
        'passed_pawns':base._passed_pawn_score}
    result={k:fn(b,True)-fn(b,False) for k,fn in functions.items()}
    assert sum(result.values())==base.evaluate_position(b)
    return result

def target():
    r=next(r for r in json.loads((ROOT/'ml/artifacts/critical-v23/fatal-and-reachable.json').read_text())['games'] if r['source_game']==15)
    return r['failure']

if __name__=='__main__':
    from engine.search import search
    from ml.train_critical_v23 import board
    f=target()
    for n,e in [('original',Evaluator()),('confinement',Evaluator(220))]:
        for depth in (2,3,4):
            p=search(board(f),depth=depth,eval_fn=e,use_lmr=True)
            print(n,depth,p.move,p.score,flush=True)
