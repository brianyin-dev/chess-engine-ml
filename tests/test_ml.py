import importlib.util
from pathlib import Path
import tempfile
import unittest

import chess

from ml.generate_data import split_for_game


HAS_TORCH = importlib.util.find_spec("torch") is not None


class DataSplitTests(unittest.TestCase):
    def test_whole_games_have_disjoint_splits(self):
        self.assertEqual([split_for_game(i) for i in range(10)].count("train"), 8)
        self.assertEqual(split_for_game(8), "val")
        self.assertEqual(split_for_game(9), "test")


@unittest.skipUnless(HAS_TORCH, "ML dependencies are optional in the engine CI environment")
class NeuralEvaluatorTests(unittest.TestCase):
    def test_encoder_includes_rule_state(self):
        from ml.model import board_to_tensor

        board = chess.Board()
        baseline = board_to_tensor(board)
        board.turn = chess.BLACK
        self.assertNotEqual(baseline.tolist(), board_to_tensor(board).tolist())
        board = chess.Board()
        board.castling_rights = chess.BB_EMPTY
        self.assertNotEqual(baseline.tolist(), board_to_tensor(board).tolist())

    def test_zero_correction_preserves_heuristic_perspective(self):
        import torch

        from engine.evaluation import evaluate
        from ml.evaluator import NeuralEvaluator
        from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE

        model = ChessNet()
        for parameter in model.parameters():
            parameter.data.zero_()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero.pt"
            torch.save({"version": MODEL_VERSION, "input_size": INPUT_SIZE,
                        "score_scale": SCORE_SCALE, "state_dict": model.state_dict()}, path)
            adapter = NeuralEvaluator(path)
            for moves in ([], ["e2e4"], ["d2d4", "d7d5", "c2c4"]):
                board = chess.Board()
                for move in moves:
                    board.push_uci(move)
                self.assertEqual(adapter(board), evaluate(board))

    def test_numpy_inference_matches_torch(self):
        import torch

        from engine.evaluation import evaluate
        from ml.evaluator import NeuralEvaluator
        from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE, board_to_tensor

        torch.manual_seed(7)
        model = ChessNet()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "random.pt"
            torch.save({"version": MODEL_VERSION, "input_size": INPUT_SIZE,
                        "score_scale": SCORE_SCALE, "state_dict": model.state_dict()}, path)
            adapter = NeuralEvaluator(path)
            board = chess.Board()
            for _ in range(30):
                with torch.inference_mode():
                    reference = round(evaluate(board) + model(board_to_tensor(board)).item() * SCORE_SCALE)
                self.assertEqual(adapter(board), reference)
                if board.is_game_over():
                    break
                board.push(next(iter(board.legal_moves)))


if __name__ == "__main__":
    unittest.main()
