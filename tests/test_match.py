import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import chess
import chess.pgn

from benchmarks.legacy.search import best_move
from benchmarks.match import choose_move, opening_board, play_game, summarize


def stats():
    return {"depth": 1, "elapsed": .01, "overrun_seconds": 0, "budget_seconds": .05}


class MatchTests(unittest.TestCase):
    def test_cli_exports_and_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'match'
            command = [sys.executable, '-m', 'benchmarks.match', '--pairs', '1',
                       '--time-ms', '2', '--max-plies', '2', '--output', str(output)]
            run = subprocess.run(command, capture_output=True, text=True,
                                 cwd=Path(__file__).resolve().parents[1], timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['summary']['unfinished'], 2)
            stream = io.StringIO((output / 'games.pgn').read_text())
            for record in report['games']:
                game = chess.pgn.read_game(stream)
                self.assertFalse(game.errors)
                self.assertEqual(game.end().board().fen(), record['final_fen'])
            self.assertIsNone(chess.pgn.read_game(stream))
            original = (output / 'report.json').read_bytes()
            retry = subprocess.run(command, capture_output=True, text=True,
                                   cwd=Path(__file__).resolve().parents[1], timeout=10)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual((output / 'report.json').read_bytes(), original)

    def test_legacy_adapter_matches_original_at_completed_depth(self):
        for moves in [[], ['e2e4', 'e7e5'], ['d2d4', 'd7d5', 'c2c4']]:
            board = chess.Board()
            for move in moves:
                board.push_uci(move)
            before = board.fen(), list(board.move_stack)
            expected = best_move(board.copy(stack=True), depth=2)
            move, data = choose_move('legacy', board, 10, depth_cap=2)
            self.assertEqual(move, expected)
            self.assertEqual(data['depth'], 2)
            self.assertFalse(data['timed_out'])
            self.assertEqual((board.fen(), board.move_stack), before)

    def test_deadlines_and_board_preservation(self):
        for engine in ['legacy', 'previous', 'current']:
            board = chess.Board()
            board.push_uci('e2e4')
            before = board.fen(), list(board.move_stack)
            move, data = choose_move(engine, board, .02)
            self.assertIn(move, board.legal_moves)
            self.assertTrue(data['timed_out'])
            self.assertLess(data['elapsed'], 2)
            self.assertEqual((board.fen(), board.move_stack), before)
            move, data = choose_move(engine, board, 1e-12)
            self.assertIn(move, board.legal_moves)
            self.assertEqual(data['depth'], 0)

    def test_opening_validation(self):
        for opening in [{}, {'name':'bad', 'moves':['e2e5']},
                        {'name':'bad', 'fen':'8/8/8/8/8/8/8/8 w - - 0 1'},
                        {'name':'finished', 'fen':'7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'}]:
            with self.assertRaises(ValueError):
                opening_board(opening)

    def test_paired_colors_same_opening_and_budget(self):
        calls = []
        def select(engine, board, budget, depth):
            calls.append((engine, board.fen(), budget, depth))
            return next(iter(board.legal_moves)), stats()
        opening = {'name':'test', 'moves':['e2e4','e7e5']}
        for color in [chess.WHITE, chess.BLACK]:
            record, pgn = play_game(opening, color, .05, 2, selector=select)
            self.assertEqual(record['result'], '*')
            self.assertEqual(record['reason'], 'ply_limit')
            self.assertIsNone(record['current_result'])
            game = chess.pgn.read_game(io.StringIO(pgn))
            self.assertFalse(game.errors)
            self.assertEqual([m.uci() for m in game.mainline_moves()][:2], opening['moves'])
            self.assertEqual(game.end().board().fen(), record['final_fen'])
        self.assertEqual(calls[0][1:], calls[2][1:])
        self.assertEqual([c[0] for c in calls], ['current','legacy','legacy','current'])

    def test_mate_result_is_correct_for_both_assignments(self):
        opening = {'name':'mate', 'fen':'7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'}
        def select(*args):
            return chess.Move.from_uci('f7g7'), stats()
        for color, expected in [(chess.WHITE, 'win'), (chess.BLACK, 'loss')]:
            record, pgn = play_game(opening, color, .05, 1, selector=select)
            self.assertEqual(record['result'], '1-0')
            self.assertEqual(record['current_result'], expected)
            self.assertEqual(record['reason'], 'checkmate')
            game = chess.pgn.read_game(io.StringIO(pgn))
            self.assertTrue(game.end().board().is_checkmate())

    def test_automatic_draw_and_history_preserved(self):
        def select(engine, board, *args):
            moves = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
            return chess.Move.from_uci(moves[len(board.move_stack) % 4]), stats()
        record, _ = play_game({'name':'repeat'}, chess.WHITE, .05, 30, selector=select)
        self.assertEqual(record['reason'], 'fivefold_repetition')
        self.assertEqual(record['result'], '1/2-1/2')
        self.assertEqual(record['played_plies'], 16)

    def test_errors_and_interruptions_are_not_losses(self):
        def illegal(*args):
            return chess.Move.from_uci('e2e5'), stats()
        def crash(*args):
            raise RuntimeError('failure')
        def interrupt(*args):
            raise KeyboardInterrupt
        for select, reason in [(illegal, 'engine_error'), (crash, 'engine_error'), (interrupt, 'interrupted')]:
            record, pgn = play_game({'name':'test'}, chess.WHITE, .05, 2, selector=select)
            self.assertEqual(record['reason'], reason)
            self.assertEqual(record['result'], '*')
            self.assertIsNone(record['current_result'])
            self.assertEqual(chess.pgn.read_game(io.StringIO(pgn)).headers['Result'], '*')

    def test_summary_excludes_unfinished_and_partial_pairs(self):
        games = [
            {'pair':1, 'current_result':'win', 'reason':'checkmate', 'moves':[]},
            {'pair':1, 'current_result':'draw', 'reason':'stalemate', 'moves':[]},
            {'pair':2, 'current_result':'loss', 'reason':'checkmate', 'moves':[]},
            {'pair':2, 'current_result':None, 'reason':'ply_limit', 'moves':[]},
            {'pair':3, 'current_result':None, 'reason':'engine_error', 'moves':[]},
        ]
        result = summarize(games)
        self.assertEqual((result['wins'],result['draws'],result['losses']), (1,1,1))
        self.assertEqual(result['unfinished'], 1)
        self.assertEqual(result['errors'], 1)
        self.assertEqual(result['score_fraction_completed'], .5)
        self.assertEqual(result['score_fraction_complete_pairs'], .75)
        self.assertIsNone(summarize([])['score_fraction_completed'])


if __name__ == '__main__':
    unittest.main()
