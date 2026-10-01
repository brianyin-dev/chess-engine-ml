"""Verify source ownership, saved games and explicit PyTorch/runtime parity."""
import json
import chess
from chess import pgn
import torch
from engine.evaluation import evaluate_position
from ml.dataset import correction_factor
from ml.evaluator import NeuralEvaluator
from ml.model import board_to_tensor,SCORE_SCALE
from ml.run_leaf_speed_v20 import ART,NEW,DATA,BASE,OLD,CP,readrows,write,canonical


def aliases(folder,split):
    records=readrows(folder/f'{split}.jsonl')+readrows(folder/'pairs'/f'{split}.jsonl')
    return ({canonical(r[f]) for r in records for f in ('fen','root_fen','good_fen','bad_fen') if f in r},
            {r['game_id'] for r in records})


def main():
    sets={s:aliases(DATA,s) for s in ('train','val','test')};new={s:aliases(NEW,s) for s in sets}
    prior=set().union(*(aliases(BASE,s)[0] for s in sets))
    position_overlap={f'{a}-{b}':len(sets[a][0]&sets[b][0]) for a,b in [('train','val'),('train','test'),('val','test')]}
    game_overlap={f'{a}-{b}':len(sets[a][1]&sets[b][1]) for a,b in [('train','val'),('train','test'),('val','test')]}
    prior_overlap={s:len(new[s][0]&prior) for s in sets}
    fens={r['fen'] for s in ('val','test') for r in readrows(NEW/f'{s}.jsonl')}
    parity={}
    for name,path in [('old5',OLD),('trained5',CP)]:
        fast=NeuralEvaluator(path,.05,True,incremental=True,fast_features=True)
        errors=[]
        with torch.inference_mode():
            for fen in fens:
                board=chess.Board(fen);actual=fast.evaluate_position(board)
                tensor=board_to_tensor(board,fast.input_size)
                expected=round(evaluate_position(board)+fast.model(tensor).item()*SCORE_SCALE*correction_factor(board,.05,True))
                if expected!=actual:errors.append({'fen':fen,'torch':expected,'runtime':actual})
        parity[name]={'positions':len(fens),'integer_mismatches':errors}
    verified={}
    for folder in ('vs-heuristic','vs-old-NN','vs-heuristic-extension'):
        path=ART/folder/'games.pgn'
        if not path.exists():continue
        count=0
        with path.open() as handle:
            while (game:=pgn.read_game(handle)) is not None:
                assert not game.errors
                board=game.board()
                for move in game.mainline_moves():assert move in board.legal_moves;board.push(move)
                assert board.result(claim_draw=True)==game.headers['Result'];count+=1
        verified[folder]=count
        assert count==(80 if folder.endswith('extension') else 20)
    report={'canonical_cross_split_overlap':position_overlap,'source_group_cross_split_overlap':game_overlap,
        'new_prior_alias_overlap':prior_overlap,'explicit_torch_runtime_parity':parity,'verified_games':verified}
    write(ART/'final-audit.json',report);print(json.dumps(report,indent=2))
    assert not any(position_overlap.values()) and not any(game_overlap.values()) and not any(prior_overlap.values())
    assert all(not p['integer_mismatches'] for p in parity.values())


if __name__=='__main__':main()
