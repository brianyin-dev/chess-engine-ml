import unittest
import importlib.util
from unittest.mock import MagicMock, patch

import chess
import chess.engine

from benchmarks.analyze import loss_data, review, score_data, significance, reference_search
from benchmarks.match import UciOpponent, play_game, summarize


class AnalysisTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'ML dependencies are optional')
    def test_draw_conversion_review_requires_lost_advantage_or_missed_mate(self):
        from ml.analyze_losses import conversion_loss
        row = {'missed_forced_mate': False, 'best_score': {'cp': 200},
               'played_score': {'cp': 20}, 'cp_loss': 180}
        self.assertTrue(conversion_loss(row))
        row['played_score']['cp'] = 100
        self.assertFalse(conversion_loss(row))
        row['best_score']['cp'] = 80
        row['played_score']['cp'] = -50
        self.assertFalse(conversion_loss(row))
        row['missed_forced_mate'] = True
        self.assertTrue(conversion_loss(row))

    def test_score_perspective(self):
        score = chess.engine.PovScore(chess.engine.Cp(150), chess.WHITE)
        self.assertEqual(score_data(score, chess.BLACK), {'cp': -150, 'mate': None})

    def test_mate_is_not_fabricated_centipawn_loss(self):
        value = loss_data({'cp': 30, 'mate': None}, {'cp': None, 'mate': -3})
        self.assertIsNone(value['cp_loss'])
        self.assertTrue(value['allows_mate'])
        self.assertFalse(loss_data({'cp': None, 'mate': -6}, {'cp': None, 'mate': -3})['allows_mate'])

    def test_loss_clamps_search_noise(self):
        self.assertEqual(loss_data({'cp': 100, 'mate': None}, {'cp': 120, 'mate': None})['cp_loss'], 0)

    def test_review_uses_same_root_and_mover_perspective(self):
        board = chess.Board()
        board.push_uci('e2e4')
        best, played = chess.Move.from_uci('e7e5'), chess.Move.from_uci('a7a6')
        engine = MagicMock()
        results = [
            {'score': chess.engine.PovScore(chess.engine.Cp(0), chess.BLACK), 'pv':[best], 'depth':15},
            {'score': chess.engine.PovScore(chess.engine.Cp(200), chess.WHITE), 'pv':[played], 'depth':16}]
        before = board.fen(), list(board.move_stack)
        with patch('benchmarks.analyze.reference_search', side_effect=results) as reference:
            result = review(engine, board, played, .1)
            self.assertEqual(reference.call_args.kwargs['root_moves'], [played])
        self.assertEqual(result['cp_loss'], 200)
        self.assertEqual((board.fen(), board.move_stack), before)

    def test_best_move_needs_no_second_search(self):
        board = chess.Board()
        move = chess.Move.from_uci('e2e4')
        engine = MagicMock()
        result = {'score':chess.engine.PovScore(chess.engine.Cp(20),chess.WHITE), 'pv':[move]}
        with patch('benchmarks.analyze.reference_search', return_value=result) as reference:
            self.assertEqual(review(engine, board, move, .1)['cp_loss'], 0)
            self.assertEqual(reference.call_count, 1)

    def test_reference_discards_unfinished_bound_scores(self):
        engine = MagicMock()
        exact = {'score':chess.engine.PovScore(chess.engine.Cp(20),chess.WHITE),
                 'pv':[chess.Move.from_uci('e2e4')], 'depth':10}
        bound = {**exact, 'score':chess.engine.PovScore(chess.engine.Cp(-80),chess.WHITE),
                 'depth':11, 'upperbound':True}
        engine.analysis.return_value.__enter__.return_value = iter([exact, bound])
        self.assertEqual(reference_search(engine, chess.Board(), .1), exact)

    def test_prioritizes_losses_in_competitive_positions(self):
        critical = {'best_score':{'cp':0,'mate':None}, 'cp_loss':150,'allows_mate':False}
        already_lost = {'best_score':{'cp':-900,'mate':None}, 'cp_loss':500,'allows_mate':False}
        self.assertGreater(significance(critical),significance(already_lost))

    def test_uci_adapter_options_and_game_reset(self):
        with patch('chess.engine.SimpleEngine.popen_uci') as start:
            fake = start.return_value
            fake.play.return_value = chess.engine.PlayResult(chess.Move.from_uci('e2e4'), None,
                                                             info={'depth':12,'nodes':100})
            selector = UciOpponent('/tmp/fake-stockfish')
            old = selector.game
            selector.new_game()
            self.assertIsNot(old, selector.game)
            move, stats = selector('stockfish', chess.Board(), .1, 64)
            self.assertIn(move, chess.Board().legal_moves)
            self.assertEqual(stats['nodes'],100)
            fake.configure.assert_called_once_with({'Threads':1,'Hash':32})
            self.assertIs(fake.play.call_args.kwargs['game'], selector.game)
            selector.close()
            fake.quit.assert_called_once()

    def test_opponent_identity_in_game_and_summary(self):
        def selector(name, board, budget, depth):
            return next(iter(board.legal_moves)), {'depth':1,'elapsed':.01,'overrun_seconds':0}
        record, pgn = play_game({'name':'test'}, chess.BLACK, .1, 2, selector=selector, opponent='stockfish')
        self.assertIn('[White "stockfish"]', pgn)
        self.assertIn('stockfish', summarize([record])['timing'])
        self.assertNotIn('legacy', summarize([record])['timing'])


if __name__ == '__main__':
    unittest.main()
