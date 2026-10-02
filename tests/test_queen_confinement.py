import unittest
import chess
from benchmarks.diagnose_bishop_v25 import queen_confinement,Evaluator,components
from benchmarks.evaluation_baseline_v25 import _material_score,evaluate_position

ROOT_FEN='2k2b1r/pp1rpp2/5n2/P1p3p1/2p1PB2/1qN4P/1P1Q1PP1/R3KB1R w KQ - 0 17'

class QueenConfinementTests(unittest.TestCase):
    def test_bishop_capture_material_is_counted(self):
        b=chess.Board(ROOT_FEN)
        self.assertEqual(_material_score(b,True)-_material_score(b,False),230)
        b.push_uci('f1c4');b.push_uci('b3c4')
        self.assertEqual(_material_score(b,True)-_material_score(b,False),0)
        self.assertEqual(evaluate_position(b),77)
        self.assertEqual(sum(components(b).values()),77)

    def test_capture_releases_confined_queen(self):
        b=chess.Board(ROOT_FEN);fen=b.fen()
        self.assertEqual(queen_confinement(b,False),-220)
        self.assertEqual(b.fen(),fen)
        b.push_uci('f1c4');b.push_uci('b3c4')
        self.assertEqual(queen_confinement(b,False),0)

    def test_color_symmetry(self):
        b=chess.Board(ROOT_FEN);m=b.mirror()
        self.assertEqual(queen_confinement(b,False),queen_confinement(m,True))
        self.assertEqual(Evaluator(220)(b),-Evaluator(220)(m))

    def test_normal_undeveloped_queens_are_exempt(self):
        b=chess.Board()
        self.assertEqual(queen_confinement(b,True),0)
        self.assertEqual(queen_confinement(b,False),0)

    def test_zero_weight_matches_original(self):
        b=chess.Board(ROOT_FEN)
        self.assertEqual(Evaluator(0)(b),evaluate_position(b))
