import unittest
import chess
from engine.search import search,_Search,INF
from engine.evaluation import evaluate
from benchmarks.search_baseline_v24 import search as baseline

class LateMoveReductionTests(unittest.TestCase):
    def test_reduced_beta_cutoff_is_verified_at_full_depth(self):
        class Probe(_Search):
            def negamax(self,b,depth,alpha,beta,ply):
                # Reduced children suggest a cutoff; full-depth children refute it.
                return -100 if depth==1 else 0
        b=chess.Board();worker=Probe(b,evaluate,None,False);worker.use_lmr=True
        score=_Search.negamax(worker,b,3,-10,10,1)
        self.assertEqual(score,0)
        self.assertGreater(worker.lmr_probes,0)
        self.assertEqual(worker.lmr_probes,worker.lmr_researches)

    def test_disabled_matches_frozen_search(self):
        b=chess.Board()
        for moves in ([],['e2e4','e7e5','g1f3']):
            b=chess.Board()
            for move in moves:b.push_uci(move)
            a=baseline(b,depth=3);z=search(b,depth=3,use_lmr=False)
            self.assertEqual((a.move,a.score,a.nodes,a.qnodes),(z.move,z.score,z.nodes,z.qnodes))

    def test_reductions_execute_and_preserve_board_history(self):
        b=chess.Board();b.push_uci('e2e4');b.push_uci('e7e5')
        before=b.fen();history=list(b.move_stack)
        worker=_Search(b,evaluate,None,True);worker.use_lmr=True
        worker.negamax(b,4,-INF,INF,0)
        self.assertGreater(worker.lmr_probes,0)
        self.assertEqual(b.fen(),before);self.assertEqual(b.move_stack,history)
        self.assertIn(worker.root_candidate,b.legal_moves)

    def test_deadline_restores_board(self):
        b=chess.Board();before=b.fen()
        p=search(b,depth=64,time_limit=.01,use_lmr=True)
        self.assertIn(p.move,b.legal_moves);self.assertEqual(b.fen(),before)

    def test_flag_is_boolean(self):
        with self.assertRaises(ValueError):search(chess.Board(),use_lmr=1)
