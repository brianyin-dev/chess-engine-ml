"""Audit frozen v19 aliases, source-game ownership and runtime score parity."""
import json
import chess
from ml.run_balanced_v19 import ART,NEW,DATA,BASE,CP,OLD,REPORTS,readrows,write
from ml.generate_search_data import key
from ml.evaluator import NeuralEvaluator
from ml.dataset import correction_factor


def aliases(path,split):
    records=readrows(path/f'{split}.jsonl')+readrows(path/'pairs'/f'{split}.jsonl')
    return {key(r[f]) for r in records for f in ('fen','root_fen','good_fen','bad_fen') if f in r}, {r['game_id'] for r in records}


def main():
    new={s:aliases(NEW,s) for s in ('train','val','test')};combined={s:aliases(DATA,s) for s in new}
    prior=set().union(*(aliases(BASE,s)[0] for s in new))
    overlaps={f'{a}-{b}':len(combined[a][0]&combined[b][0]) for a,b in [('train','val'),('train','test'),('val','test')]}
    game_overlaps={f'{a}-{b}':len(combined[a][1]&combined[b][1]) for a,b in [('train','val'),('train','test'),('val','test')]}
    prior_overlap={s:len(new[s][0]&prior) for s in new}
    reviewed=set()
    for path in REPORTS:
        for game in json.loads(path.read_text())['games']:
            board=chess.Board(game['initial_fen'])
            for uci in game['opening_moves']+[m['uci'] for m in game['moves']]:
                reviewed.add(key(board.fen()));board.push_uci(uci)
            reviewed.add(key(board.fen()))
    review_overlap={s:len(new[s][0]&reviewed) for s in ('val','test')}
    frozen=json.loads((ART/'frozen-selection.json').read_text());w=frozen['weight']
    evaluators=[NeuralEvaluator(CP,w,True,incremental=True),NeuralEvaluator(CP,w,True),NeuralEvaluator(CP,w,True,optimized=False)]
    fens={r['fen'] for s in ('val','test') for r in readrows(NEW/f'{s}.jsonl')};mismatches=[]
    for fen in fens:
        board=chess.Board(fen);scores=[e.evaluate_position(board) for e in evaluators]
        if len(set(scores))!=1:mismatches.append({'fen':fen,'scores':scores})
    attainability={}
    for weight in (.05,.1,.25):
        attainability[str(weight)]={}
        for split in ('val','test'):
            hard=[r for r in readrows(NEW/'pairs'/f'{split}.jsonl') if r['heuristic_gap_cp']<=0]
            possible=sum(r['heuristic_gap_cp']+250*sum(correction_factor(chess.Board(r[f]),weight,True) for f in ('good_fen','bad_fen'))>=5 for r in hard)
            attainability[str(weight)][split]={'hard_pairs':len(hard),'potentially_attainable_5cp_margin':possible}
    import torch
    previous=torch.load(OLD,map_location='cpu',weights_only=True)['state_dict']
    chosen=torch.load(CP,map_location='cpu',weights_only=True)['state_dict']
    state_equal=previous.keys()==chosen.keys() and all(torch.equal(previous[k],chosen[k]) for k in previous)
    from chess import pgn
    pgn_counts={}
    for name in ('vs-heuristic','vs-old-NN'):
        count=0
        with (ART/name/'games.pgn').open() as handle:
            while (game:=pgn.read_game(handle)) is not None:
                assert not game.errors
                board=game.board()
                for move in game.mainline_moves():
                    assert move in board.legal_moves;board.push(move)
                assert board.result(claim_draw=True)==game.headers['Result']
                count+=1
        pgn_counts[name]=count
        assert count==20
    result={'checkpoint_parameters_equal_initial_v8':state_equal,'verified_pgn_counts':pgn_counts,'bounded_correction_attainability':attainability,'canonical_cross_split_overlap':overlaps,'source_game_cross_split_overlap':game_overlaps,
        'new_prior_alias_overlap':prior_overlap,'holdout_reviewed_game_alias_overlap':review_overlap,
        'inference_positions':len(fens),'inference_mismatches':mismatches}
    write(ART/'integrity-and-parity.json',result);print(json.dumps(result,indent=2))
    assert not any(overlaps.values()) and not any(game_overlaps.values()) and not any(prior_overlap.values()) and not mismatches


if __name__=='__main__':main()
