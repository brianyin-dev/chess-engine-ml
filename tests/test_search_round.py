import random
import unittest
from unittest.mock import patch
import chess
from engine.search import search, _Search, _SearchLimit, _tactical_moves, INF
from engine.evaluation import evaluate
from benchmarks.pre_round.search import search as old_search
from benchmarks.pre_round.evaluation import evaluate as old_evaluate

DUTCH = 'rn1q1rk1/pp1bb2p/4p3/2ppNpp1/3PnB2/2NQP1P1/PPP2PBP/R4RK1 w - - 0 11'


class SearchRoundTests(unittest.TestCase):
    def test_tactical_generation_matches_full_legal_filter(self):
        rng = random.Random(42)
        boards = [chess.Board(), chess.Board('1r5k/P7/8/8/8/8/8/7K w - - 0 1'),
                  chess.Board('7k/8/8/3pP3/8/8/8/K7 w - d6 0 1')]
        game = chess.Board()
        for _ in range(150):
            if game.is_game_over():
                game = chess.Board()
            game.push(rng.choice(list(game.legal_moves)))
            boards.append(game.copy())
        for board in boards:
            for b in [board, board.mirror()]:
                expected = {m for m in b.legal_moves if b.is_capture(m) or m.promotion}
                generated = _tactical_moves(b)
                self.assertEqual(set(generated), expected)
                self.assertEqual(len(generated), len(set(generated)))

    def test_evaluation_scores_unchanged(self):
        board, rng = chess.Board(), random.Random(27)
        for _ in range(80):
            self.assertEqual(evaluate(board), old_evaluate(board))
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))

    def test_fixed_depth_scores_match_previous_search(self):
        for fen in [chess.STARTING_FEN, '3r3k/8/8/3p4/8/8/8/K2Q4 w - - 0 1',
                    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                    '7k/8/8/3pP3/8/8/8/K7 w - d6 0 1']:
            board = chess.Board(fen)
            self.assertEqual(search(board, depth=2).score, old_search(board, depth=2).score)

    def test_scored_fallback_when_first_iteration_interrupted(self):
        board = chess.Board(DUTCH)
        before = board.fen(), list(board.move_stack)
        with patch.object(_Search, 'negamax', side_effect=_SearchLimit('time_limit')):
            result = search(board, time_limit=1)
        self.assertEqual(result.depth, 0)
        self.assertIsNone(result.score)
        self.assertGreater(result.fallback_evaluations, 0)
        self.assertNotEqual(result.move.uci(), 'e5f7')
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual((board.fen(), board.move_stack), before)

    def test_fallback_recognizes_mate(self):
        board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
        worker = _Search(board, evaluate, None, True)
        move = worker.fallback(board, list(board.legal_moves), None)
        board.push(move)
        self.assertTrue(board.is_checkmate())

    def test_static_cache_does_not_override_draw_rules(self):
        board = chess.Board('7k/8/8/8/8/8/8/KQ6 w - - 0 1')
        worker = _Search(board, evaluate, None, True)
        value = worker.static(board)
        self.assertEqual(worker.static(board), value)
        self.assertEqual(worker.static_cache_hits, 1)
        board.halfmove_clock = 150
        self.assertEqual(worker.quiescence(board, -INF, INF, 0), 0)

    def test_custom_evaluators_are_not_cached(self):
        board = chess.Board()
        calls = []
        def evaluate_custom(b):
            calls.append(b.fen())
            return len(calls)
        worker = _Search(board, evaluate_custom, None, True)
        self.assertEqual(worker.static(board), 1)
        self.assertEqual(worker.static(board), 2)
        self.assertEqual(worker.static_cache_hits, 0)

    def test_declared_position_only_evaluator_caches_with_clock(self):
        class PositionEvaluator:
            cacheable_by_fen = True

            def __init__(self):
                self.calls = 0

            def __call__(self, board):
                self.calls += 1
                return board.halfmove_clock

        board = chess.Board()
        evaluator = PositionEvaluator()
        worker = _Search(board, evaluator, None, True)
        self.assertEqual(worker.static(board), 0)
        self.assertEqual(worker.static(board), 0)
        self.assertEqual(evaluator.calls, 1)
        board.halfmove_clock = 1
        self.assertEqual(worker.static(board), 1)
        self.assertEqual(evaluator.calls, 2)

    def test_stalemate_in_quiescence(self):
        board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')
        worker = _Search(board, evaluate, None, True)
        self.assertEqual(worker.quiescence(board, -INF, INF, 0), 0)


if __name__ == '__main__':
    unittest.main()
