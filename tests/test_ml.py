import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
    def test_candidate_ranking_uses_root_mover_perspective(self):
        import json
        from ml.model import ChessNet
        from ml.train import pair_loader, ranking_accuracy

        bad = chess.Board()
        good = bad.copy()
        good.remove_piece_at(chess.D8)
        model = ChessNet()
        for p in model.parameters():
            p.data.zero_()
        rows = [{'good_fen': good.fen(), 'bad_fen': bad.fen(), 'sign': 1},
                {'good_fen': good.mirror().fen(), 'bad_fen': bad.mirror().fen(), 'sign': -1}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertEqual(ranking_accuracy(model, pair_loader(path, False)), 1)

    def test_material_features_and_phase(self):
        from ml.model import board_to_tensor, material_score

        board = chess.Board()
        initial = board_to_tensor(board)
        self.assertEqual(initial[792].item(), 0)
        self.assertEqual(initial[793].item(), 1)
        board.remove_piece_at(chess.D8)
        features = board_to_tensor(board)
        self.assertEqual(material_score(board), 900)
        self.assertAlmostEqual(features[792].item(), .225)
        self.assertLess(features[793].item(), initial[793].item())
        self.assertEqual(material_score(board.mirror()), -900)

    def test_material_anchor_bounds_neural_correction(self):
        import torch
        from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE, material_score, board_to_tensor
        from ml.evaluator import NeuralEvaluator

        model = ChessNet(correction_limit_cp=250)
        for p in model.parameters():
            p.data.zero_()
        model.net[-1].bias.data.fill_(-100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'material.pt'
            torch.save({'version': MODEL_VERSION, 'input_size': INPUT_SIZE,
                        'score_scale': SCORE_SCALE, 'target_mode': 'material',
                        'correction_limit_cp': 250, 'state_dict': model.state_dict()}, path)
            evaluator = NeuralEvaluator(path)
            board = chess.Board()
            board.remove_piece_at(chess.D8)
            with torch.inference_mode():
                reference = round(material_score(board) + model(board_to_tensor(board)).item() * SCORE_SCALE)
            self.assertEqual(evaluator(board), reference)
            self.assertGreaterEqual(evaluator(board), 650)

    def test_legacy_checkpoint_remains_loadable(self):
        import torch
        from ml.model import ChessNet, LEGACY_INPUT_SIZE, SCORE_SCALE
        from ml.evaluator import NeuralEvaluator

        model = ChessNet(LEGACY_INPUT_SIZE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save({'version': 2, 'input_size': LEGACY_INPUT_SIZE,
                        'score_scale': SCORE_SCALE, 'state_dict': model.state_dict()}, path)
            self.assertIsInstance(NeuralEvaluator(path)(chess.Board()), int)

    def test_target_modes_encode_full_score_and_residual(self):
        from engine.evaluation import evaluate
        from ml.dataset import ChessEvalDataset
        from ml.model import SCORE_SCALE

        board = chess.Board()
        board.push_uci("e2e4")
        label = 75
        row = {"game_id": 0, "fen": board.fen(), "score_cp": label}
        absolute = ChessEvalDataset([row], "absolute")
        residual = ChessEvalDataset([row], "residual")
        self.assertAlmostEqual(absolute.targets[0].item() * SCORE_SCALE, label, places=4)
        self.assertAlmostEqual(residual.targets[0].item() * SCORE_SCALE,
                               label - evaluate(board), places=4)

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

    def test_absolute_mode_does_not_call_heuristic(self):
        import torch

        from ml.evaluator import NeuralEvaluator
        from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE

        model = ChessNet()
        for parameter in model.parameters():
            parameter.data.zero_()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absolute.pt"
            torch.save({"version": MODEL_VERSION, "input_size": INPUT_SIZE,
                        "score_scale": SCORE_SCALE, "target_mode": "absolute",
                        "state_dict": model.state_dict()}, path)
            adapter = NeuralEvaluator(path)
            board = chess.Board()
            board.push_uci("e2e4")
            with patch("ml.evaluator.evaluate", side_effect=AssertionError("heuristic called")):
                self.assertEqual(adapter(board), 0)

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
