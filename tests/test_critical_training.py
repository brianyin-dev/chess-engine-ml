import unittest
import chess
from ml.diagnose_critical_v23 import critical,forced,Trace
from ml.evaluator import NeuralEvaluator
from ml.run_search_v22 import OLD
from engine.search import _Search,INF
from ml.confirm_fatal_v23 import fatal

class CriticalMistakeTests(unittest.TestCase):
    def test_fatal_flag_requires_avoidable_losing_state(self):
        def flag(best,played):
            return fatal({'best_score':best,'played_score':played,**critical(best,played)})
        self.assertTrue(flag({'cp':0,'mate':None},{'cp':-150,'mate':None}))
        self.assertFalse(flag({'cp':-300,'mate':None},{'cp':-500,'mate':None}))
        self.assertFalse(flag({'cp':300,'mate':None},{'cp':100,'mate':None}))
        self.assertTrue(flag({'cp':0,'mate':None},{'cp':None,'mate':-3}))
    def test_minus150_crossing_requires_real_deterioration(self):
        best={'cp':0,'mate':None};bad={'cp':-150,'mate':None}
        self.assertTrue(critical(best,bad)['qualifies'])
        self.assertTrue(critical(best,bad)['crosses_minus_150'])
        self.assertFalse(critical({'cp':-300,'mate':None},{'cp':-400,'mate':None})['qualifies'])
        self.assertFalse(critical({'cp':-140,'mate':None},{'cp':-151,'mate':None})['qualifies'])

    def test_forced_trace_agrees_with_full_window_search_and_restores_history(self):
        m=NeuralEvaluator(OLD,0,True,incremental=True,fast_features=True)
        b=chess.Board();b.push_uci('e2e4');b.push_uci('e7e5');fen=b.fen();history=list(b.move_stack)
        move=chess.Move.from_uci('g1f3');trace=forced(b,move,m,depth=2)
        w=_Search(b,m,None,False)
        with w.pushed(b,move):expected=-w.negamax(b,1,-INF,INF,1)
        self.assertEqual(trace['root_score_cp'],expected)
        self.assertEqual(b.fen(),fen);self.assertEqual(b.move_stack,history)
