import random
import unittest

import chess

from benchmarks.run import check_move, load_positions
from engine.evaluation import evaluate, _passed_pawn_score, _pst_score, MATE_SCORE
from engine.search import search, _Search, _to_tt, _from_tt


class EvaluationTests(unittest.TestCase):
    def test_starting_position_is_balanced(self):
        self.assertEqual(evaluate(chess.Board()), 0)

    def test_central_pawn_development_is_rewarded(self):
        central, flank = chess.Board(), chess.Board()
        central.push_uci("e2e4")
        flank.push_uci("h2h3")
        self.assertGreater(evaluate(central), evaluate(flank))

    def test_piece_tables_reward_seventh_rank_pawn(self):
        second = chess.Board("7k/8/8/8/8/8/P7/7K w - - 0 1")
        seventh = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        self.assertGreater(_pst_score(seventh, chess.WHITE), _pst_score(second, chess.WHITE))

    def test_passed_pawn_bonus_increases_toward_promotion(self):
        scores = []
        for rank in range(1, 7):
            board = chess.Board("7k/8/8/8/8/8/8/7K w - - 0 1")
            board.set_piece_at(chess.square(0, rank), chess.Piece(chess.PAWN, chess.WHITE))
            scores.append(_passed_pawn_score(board, chess.WHITE))
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])))

    def test_passed_pawn_masks_match_reference(self):
        from engine.evaluation import PASSED_PAWN_BONUS
        rng = random.Random(19)
        for _ in range(40):
            board = chess.Board.empty()
            for square in rng.sample(list(range(8, 56)), 16):
                board.set_piece_at(square, chess.Piece(chess.PAWN, rng.choice(chess.COLORS)))
            for color in chess.COLORS:
                expected = 0
                for square in board.pieces(chess.PAWN, color):
                    rank = chess.square_rank(square)
                    blockers = [enemy for enemy in board.pieces(chess.PAWN, not color)
                                if abs(chess.square_file(enemy) - chess.square_file(square)) <= 1
                                and (chess.square_rank(enemy) > rank if color else chess.square_rank(enemy) < rank)]
                    if not blockers:
                        expected += PASSED_PAWN_BONUS[rank if color else 7 - rank]
                self.assertEqual(_passed_pawn_score(board, color), expected)

    def test_king_prefers_activity_in_endgame(self):
        corner = chess.Board("7k/7p/8/8/8/8/P7/K7 w - - 0 1")
        center = chess.Board("7k/7p/8/8/3K4/8/P7/8 w - - 0 1")
        self.assertGreater(evaluate(center), evaluate(corner))

    def test_king_prefers_shelter_with_full_material(self):
        sheltered = chess.Board()
        exposed = sheltered.copy()
        exposed.remove_piece_at(chess.E1)
        exposed.set_piece_at(chess.E4, chess.Piece(chess.KING, chess.WHITE))
        self.assertGreater(_pst_score(sheltered, chess.WHITE), _pst_score(exposed, chess.WHITE))

    def test_color_symmetry_and_no_mutation(self):
        board, rng = chess.Board(), random.Random(42)
        for _ in range(50):
            before = (board.fen(), list(board.move_stack))
            self.assertEqual(evaluate(board), -evaluate(board.mirror()))
            self.assertEqual((board.fen(), board.move_stack), before)
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))

    def test_automatic_draw_scores_zero(self):
        board = chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 150 90")
        self.assertEqual(evaluate(board), 0)


class SearchTests(unittest.TestCase):
    def test_tactical_regressions(self):
        for case in load_positions():
            if not any(k in case for k in ("expect", "best", "avoid")):
                continue
            with self.subTest(case=case["id"]):
                board = chess.Board(case["fen"])
                result = search(board, depth=1)
                self.assertTrue(check_move(case, board, result.move))
                # Mirrored colors must solve the same tactical problem.
                mirrored = board.mirror()
                mirror_case = dict(case)
                for key in ("best", "avoid"):
                    if key in case:
                        mirror_case[key] = []
                        for uci in case[key]:
                            m = chess.Move.from_uci(uci)
                            mirror_case[key].append(chess.Move(
                                chess.square_mirror(m.from_square), chess.square_mirror(m.to_square),
                                promotion=m.promotion).uci())
                self.assertTrue(check_move(mirror_case, mirrored, search(mirrored, depth=1).move))

    def test_mate_is_rule_based_even_with_bad_evaluator(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = search(board, depth=2, eval_fn=lambda _: -999999)
        board.push(result.move)
        self.assertTrue(board.is_checkmate())
        self.assertEqual(result.score, MATE_SCORE - 1)

    def test_checkmate_and_stalemate_return_no_move(self):
        for fen, score in [
            ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", -MATE_SCORE),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0),
            ("7k/8/8/8/8/8/8/K7 w - - 0 1", 0),
        ]:
            result = search(chess.Board(fen))
            self.assertIsNone(result.move)
            self.assertEqual(result.score, score)

    def test_timeout_returns_legal_move_and_restores_board(self):
        board = chess.Board()
        board.push_uci("e2e4")
        before = (board.fen(), list(board.move_stack))
        result = search(board, depth=64, time_limit=0.03)
        self.assertTrue(result.timed_out)
        self.assertIn(result.move, board.legal_moves)
        self.assertLess(result.elapsed, 2)
        self.assertEqual((board.fen(), board.move_stack), before)

    def test_tiny_budget_returns_fallback(self):
        board = chess.Board()
        result = search(board, time_limit=1e-12)
        self.assertEqual(result.depth, 0)
        self.assertIsNone(result.score)
        self.assertIn(result.move, board.legal_moves)

    def test_evaluator_error_restores_history(self):
        board = chess.Board()
        board.push_uci("e2e4")
        before = (board.fen(), list(board.move_stack))
        def broken(_):
            raise RuntimeError("model failed")
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            search(board, eval_fn=broken)
        self.assertEqual((board.fen(), board.move_stack), before)

    def test_quiescence_never_stands_pat_in_check(self):
        board = chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 0 1")
        def evaluator(position):
            self.assertFalse(position.is_check())
            return evaluate(position)
        search(board, depth=1, eval_fn=evaluator)

    def test_special_positions_survive_search(self):
        for fen in ["r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                    "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
                    "7k/P7/8/8/8/8/8/7K w - - 0 1"]:
            board = chess.Board(fen)
            result = search(board, depth=2)
            self.assertIn(result.move, board.legal_moves)
            self.assertEqual(board.fen(), fen)
            self.assertEqual(board.move_stack, [])

    def test_cache_preserves_scores(self):
        for fen in [chess.STARTING_FEN, "3r3k/8/8/3p4/8/8/8/K2Q4 w - - 0 1"]:
            board = chess.Board(fen)
            self.assertEqual(search(board, depth=3, use_tt=True).score,
                             search(board, depth=3, use_tt=False).score)

    def test_cache_distinguishes_draw_context(self):
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            board.push_uci(uci)
        no_history = chess.Board(board.fen())
        a, b = _Search(board, evaluate, None, True), _Search(no_history, evaluate, None, True)
        self.assertNotEqual(a.cache_key(board), b.cache_key(no_history))
        changed_clock = chess.Board(board.fen())
        changed_clock.halfmove_clock += 1
        self.assertNotEqual(b.cache_key(no_history), b.cache_key(changed_clock))
        # Threefold is claimable, not automatic; this app has no claim action.
        self.assertIsNotNone(search(board, depth=1).move)
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            board.push_uci(uci)
        self.assertTrue(board.is_fivefold_repetition())
        result = search(board)
        self.assertIsNone(result.move)
        self.assertEqual(result.score, 0)

    def test_mate_distance_cache_normalization(self):
        for score in (MATE_SCORE - 8, -MATE_SCORE + 8, 42):
            self.assertEqual(_from_tt(_to_tt(score, 3), 3), score)
        self.assertGreater(_from_tt(_to_tt(MATE_SCORE - 8, 3), 1), MATE_SCORE - 8)

    def test_bad_limits_rejected(self):
        for depth in [0, -1, 65, True, 1.5, "3"]:
            with self.assertRaises(ValueError):
                search(chess.Board(), depth=depth)
        for budget in [0, -1, float("nan"), float("inf"), True, "1"]:
            with self.assertRaises(ValueError):
                search(chess.Board(), time_limit=budget)


if __name__ == "__main__":
    unittest.main()
