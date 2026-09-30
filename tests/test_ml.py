import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess

from ml.generate_data import split_for_game


HAS_TORCH = importlib.util.find_spec("torch") is not None


class DataSplitTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "ML dependencies are optional")
    def test_search_split_key_groups_color_mirrors_and_move_counters(self):
        from ml.generate_search_data import key
        board = chess.Board()
        board.push_uci('e2e4')
        self.assertEqual(key(board.fen()), key(board.mirror().fen()))
        alternate = board.copy()
        alternate.fullmove_number = 42
        alternate.halfmove_clock = 3
        self.assertEqual(key(board.fen()), key(alternate.fen()))
        alternate.push_uci('c7c5')
        self.assertNotEqual(key(board.fen()), key(alternate.fen()))

    def test_whole_games_have_disjoint_splits(self):
        self.assertEqual([split_for_game(i) for i in range(10)].count("train"), 8)
        self.assertEqual(split_for_game(8), "val")
        self.assertEqual(split_for_game(9), "test")


@unittest.skipUnless(HAS_TORCH, "ML dependencies are optional in the engine CI environment")
class NeuralEvaluatorTests(unittest.TestCase):
    def test_zero_weight_benchmark_keeps_optimized_baseline_for_both_players(self):
        import json
        import io
        from contextlib import redirect_stdout
        import torch
        from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE
        from ml.compare import main
        from engine.search import search
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'network.pt';openings=root/'openings.json';output=root/'match'
            model=ChessNet(INPUT_SIZE,250,True)
            torch.save({'version':MODEL_VERSION,'input_size':INPUT_SIZE,'score_scale':SCORE_SCALE,
                        'target_mode':'residual','correction_limit_cp':250,'color_consistent':True,
                        'state_dict':model.state_dict()},checkpoint)
            openings.write_text(json.dumps([{'name':'start','moves':[]}]))
            args=['compare','--checkpoint',str(checkpoint),'--openings',str(openings),'--output',str(output),
                  '--pairs','1','--max-plies','2','--nodes','128','--nn-weight','0','--quiet-only','--incremental',
                  '--opponent-checkpoint',str(checkpoint),'--opponent-nn-weight','0',
                  '--opponent-quiet-only','--opponent-incremental']
            with patch('sys.argv',args),patch('ml.compare.search',wraps=search) as calls,redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(calls.call_count,4)
            for call in calls.call_args_list:
                evaluator=call.kwargs['eval_fn']
                self.assertTrue(evaluator.incremental)
                self.assertEqual(evaluator.correction_weight,0)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['candidate'],'incremental-heuristic')
            self.assertTrue(report['opponent']['incremental'])

    def test_incremental_features_survive_special_moves_and_undo(self):
        import numpy as np
        import random
        from ml.incremental import IncrementalEncoder
        from ml.model import board_to_array
        from ml.incremental import IncrementalBaseline
        from engine.evaluation import evaluate_position
        encoder=IncrementalEncoder(892)
        baseline=IncrementalBaseline()
        for fen, move in [('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','e1g1'),
                          ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2','e5d6'),
                          ('1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1','a7b8n')]:
            board=chess.Board(fen)
            for action in ('initial','push','pop'):
                if action=='push':board.push_uci(move)
                elif action=='pop':board.pop()
                np.testing.assert_array_equal(encoder.encode(board),board_to_array(board,892))
                self.assertEqual(baseline.evaluate(board),evaluate_position(board))
        rng=random.Random(15);board=chess.Board()
        for _ in range(300):
            np.testing.assert_array_equal(encoder.encode(board),board_to_array(board,892))
            self.assertEqual(baseline.evaluate(board),evaluate_position(board))
            if board.is_game_over():board=chess.Board()
            elif board.move_stack and rng.random()<.2:board.pop()
            else:board.push(rng.choice(list(board.legal_moves)))

    def test_incremental_search_preserves_neural_scores_and_decisions(self):
        import torch
        from ml.model import ChessNet, RELATIONAL_INPUT_SIZE, RELATIONAL_MODEL_VERSION, SCORE_SCALE
        from ml.evaluator import NeuralEvaluator
        from engine.search import search
        torch.manual_seed(15)
        model=ChessNet(RELATIONAL_INPUT_SIZE,250,True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'network.pt'
            torch.save({'version':RELATIONAL_MODEL_VERSION,'input_size':RELATIONAL_INPUT_SIZE,
                        'score_scale':SCORE_SCALE,'target_mode':'residual','correction_limit_cp':250,
                        'color_consistent':True,'state_dict':model.state_dict()},path)
            old=NeuralEvaluator(path,.25,quiet_only=True)
            new=NeuralEvaluator(path,.25,quiet_only=True,incremental=True)
            for board in (chess.Board(),chess.Board().mirror()):
                self.assertEqual(old.evaluate_position(board),new.evaluate_position(board))
                a=search(board,depth=2,eval_fn=old);b=search(board,depth=2,eval_fn=new)
                self.assertEqual((a.move,a.score,a.nodes,a.qnodes),(b.move,b.score,b.nodes,b.qnodes))

    def test_compact_projection_preserves_selected_connections(self):
        import torch
        from ml.model import ChessNet, RELATIONAL_INPUT_SIZE, RELATIONAL_MODEL_VERSION, INPUT_SIZE
        from ml.project_compact import project
        model = ChessNet(RELATIONAL_INPUT_SIZE, 250, True)
        saved = {'version': RELATIONAL_MODEL_VERSION, 'input_size': RELATIONAL_INPUT_SIZE,
                 'target_mode': 'residual', 'correction_limit_cp': 250, 'color_consistent': True,
                 'state_dict': model.state_dict()}
        result, report = project(saved)
        first, second = report['first_units'], report['second_units']
        self.assertEqual(result['input_size'], INPUT_SIZE)
        self.assertTrue(torch.equal(result['state_dict']['net.0.weight'], saved['state_dict']['net.0.weight'][first,:INPUT_SIZE]))
        self.assertTrue(torch.equal(result['state_dict']['net.2.weight'], saved['state_dict']['net.2.weight'][second][:,first]))
        with self.assertRaises(ValueError):
            project(saved, (65,16))

    def test_compact_checkpoint_inference_matches_torch(self):
        import torch
        from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE, board_to_tensor
        from ml.evaluator import NeuralEvaluator
        from engine.evaluation import evaluate
        torch.manual_seed(14)
        model = ChessNet(INPUT_SIZE, 250, True, (32, 16)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'compact.pt'
            torch.save({'version': MODEL_VERSION, 'input_size': INPUT_SIZE, 'score_scale': SCORE_SCALE,
                        'target_mode': 'residual', 'hidden_sizes': [32,16], 'correction_limit_cp': 250,
                        'color_consistent': True, 'state_dict': model.state_dict()}, path)
            evaluator = NeuralEvaluator(path, .25)
            board = chess.Board()
            for position in (board, board.mirror()):
                with torch.inference_mode():
                    predicted = round(evaluate(position) + model(board_to_tensor(position)).item() * SCORE_SCALE * .25)
                self.assertEqual(evaluator(position), predicted)
            self.assertEqual(evaluator(board), -evaluator(board.mirror()))
        with self.assertRaises(ValueError):
            ChessNet(hidden_sizes=(32, 0))

    def test_rounded_training_scores_match_full_runtime_and_keep_gradient(self):
        import torch
        from ml.train import rounded_score
        correction = torch.tensor([.5 / 400, -.5 / 400, .4 / 400], requires_grad=True)
        baseline = torch.tensor([1., 1., 0.])
        result = rounded_score(baseline, correction, True)
        self.assertEqual(result.tolist(), [2., 0., 0.])
        result.sum().backward()
        self.assertEqual(correction.grad.tolist(), [400., 400., 400.])

    def test_validation_ranking_uses_integer_search_scores(self):
        import torch
        from ml.train import ranking_accuracy
        class SubCentipawnModel(torch.nn.Module):
            def forward(self, x):
                return x[:, 0] / 400
        # A positive float margin becomes a tie after runtime rounding.
        tensors = (torch.tensor([[.4]]), torch.tensor([[0.]]), torch.tensor([0.]),
                   torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.]))
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors))
        self.assertEqual(ranking_accuracy(SubCentipawnModel(), loader), 0.)

    def test_pawn_relationship_masks_match_geometric_definition(self):
        from ml.model import relationship_features
        import random
        rng = random.Random(1200250)
        board = chess.Board()
        for _ in range(120):
            values = relationship_features(board)
            for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
                direction = 1 if color else -1
                pawns = list(board.pieces(chess.PAWN, color))
                enemy = list(board.pieces(chess.PAWN, not color))
                king = board.king(color)
                passed = sum(not any(abs(chess.square_file(p) - chess.square_file(e)) <= 1
                                     and direction * (chess.square_rank(e) - chess.square_rank(p)) > 0
                                     for e in enemy) for p in pawns)
                shield = sum(king is not None and abs(chess.square_file(p) - chess.square_file(king)) <= 1
                             and 0 < direction * (chess.square_rank(p) - chess.square_rank(king)) <= 2
                             for p in pawns)
                self.assertEqual(values[color_index * 49 + 47], passed / 8)
                self.assertEqual(values[color_index * 49 + 48], shield / 8)
            if board.is_game_over():
                board = chess.Board()
            else:
                board.push(rng.choice(list(board.legal_moves)))

    def test_mistake_importance_is_training_only(self):
        import json
        from ml.train import pair_loader
        board = chess.Board()
        row = {'good_fen': board.fen(), 'bad_fen': board.mirror().fen(),
               'sign': 1, 'training_weight': 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.jsonl'
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual(next(iter(pair_loader(path, True)))[-1].item(), 4)
            self.assertEqual(next(iter(pair_loader(path, False)))[-1].item(), 1)
            row['training_weight'] = -1
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                pair_loader(path, True)

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

@unittest.skipUnless(HAS_TORCH, 'ML dependencies are optional')
class ColorConsistencyTests(unittest.TestCase):
    def test_mirror_inference_parity_and_material_direction(self):
        import torch
        from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE, board_to_tensor, material_score
        from ml.evaluator import NeuralEvaluator
        torch.manual_seed(461)
        model = ChessNet(correction_limit_cp=45, color_consistent=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'symmetric.pt'
            torch.save({'version': MODEL_VERSION, 'input_size': INPUT_SIZE,
                        'score_scale': SCORE_SCALE, 'target_mode': 'material',
                        'correction_limit_cp': 45, 'color_consistent': True,
                        'state_dict': model.state_dict()}, path)
            evaluator = NeuralEvaluator(path)
            boards = [chess.Board()]
            for move in ('e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4'):
                board = boards[-1].copy()
                board.push_uci(move)
                boards.append(board)
            for board in boards + [b.mirror() for b in boards]:
                self.assertEqual(evaluator(board), -evaluator(board.mirror()))
                with torch.inference_mode():
                    expected = round(material_score(board) + model(board_to_tensor(board)).item() * SCORE_SCALE)
                self.assertEqual(evaluator(board), expected)
                for color in chess.COLORS:
                    for square in board.pieces(chess.PAWN, color):
                        changed = board.copy()
                        changed.remove_piece_at(square)
                        sign = 1 if color == chess.BLACK else -1
                        self.assertGreater(sign * (evaluator(changed) - evaluator(board)), 0)
            features = torch.stack([board_to_tensor(b) for b in boards])
            mirrors = torch.stack([board_to_tensor(b.mirror()) for b in boards])
            self.assertTrue(torch.equal(model(features), -model(mirrors)))

@unittest.skipUnless(HAS_TORCH, 'ML dependencies are optional')
class RelationshipEvaluatorTests(unittest.TestCase):
    def test_relational_features_color_mirror_and_inference_parity(self):
        import torch
        import numpy as np
        from ml.model import (ChessNet, RELATIONAL_INPUT_SIZE, RELATIONAL_MODEL_VERSION,
                              SCORE_SCALE, board_to_tensor, board_to_array)
        from ml.evaluator import NeuralEvaluator
        from engine.evaluation import evaluate
        torch.manual_seed(719)
        model = ChessNet(RELATIONAL_INPUT_SIZE, 200, True)
        boards = [chess.Board(), chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),
                  chess.Board('r3k2r/pp3ppp/8/8/8/8/PP3PPP/R3K2R b KQkq - 0 1')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'relationships.pt'
            torch.save({'version': RELATIONAL_MODEL_VERSION, 'input_size': RELATIONAL_INPUT_SIZE,
                        'score_scale': SCORE_SCALE, 'target_mode': 'residual',
                        'correction_limit_cp': 200, 'color_consistent': True,
                        'state_dict': model.state_dict()}, path)
            evaluator = NeuralEvaluator(path)
            for board in boards:
                x = board_to_array(board, RELATIONAL_INPUT_SIZE)[794:].reshape(2,49)
                mirror = board_to_array(board.mirror(), RELATIONAL_INPUT_SIZE)[794:].reshape(2,49)
                np.testing.assert_array_equal(x, mirror[::-1])
                for b in (board, board.mirror()):
                    self.assertEqual(evaluator(b), -evaluator(b.mirror()))
                    with torch.inference_mode():
                        expected = round(evaluate(b) + model(board_to_tensor(b,RELATIONAL_INPUT_SIZE)).item()*SCORE_SCALE)
                    self.assertEqual(evaluator(b), expected)

    def test_attacked_undefended_feature_changes_with_defender(self):
        from ml.model import relationship_features
        hanging = chess.Board('3r3k/8/8/3Q4/8/8/8/K7 b - - 0 1')
        defended = hanging.copy()
        defended.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        before, after = relationship_features(hanging), relationship_features(defended)
        # White queen attack/defense/undefended is the final three piece-count features.
        self.assertEqual(before[44], 1/8)
        self.assertEqual(after[44], 0)
        self.assertEqual(after[43], 1/8)

@unittest.skipUnless(HAS_TORCH, 'ML dependencies are optional')
class WeightedCorrectionTests(unittest.TestCase):
    def test_training_blend_matches_runtime_for_quiet_and_tactical_pairs(self):
        import json
        import torch
        from ml.train import pair_loader
        from ml.evaluator import NeuralEvaluator
        from ml.dataset import ChessEvalDataset
        from ml.model import SCORE_SCALE
        capture = chess.Board()
        capture.push_uci('e2e4'); capture.push_uci('d7d5')
        quiet = chess.Board()
        with tempfile.TemporaryDirectory() as directory:
            evaluator = NeuralEvaluator(self.checkpoint(directory), .25, quiet_only=True)
            path = Path(directory) / 'pairs.jsonl'
            path.write_text(json.dumps({'good_fen': capture.fen(), 'bad_fen': quiet.fen(), 'sign': 1}) + '\n')
            good, bad, base, sign, good_factor, bad_factor, _ = next(iter(
                pair_loader(path, False, target_mode='residual', correction_weight=.25, quiet_only=True)))
            self.assertEqual(good_factor.item(), 0)
            self.assertEqual(bad_factor.item(), .25)
            with torch.inference_mode():
                gap = sign * (good_factor * evaluator.model(good) - bad_factor * evaluator.model(bad) + base) * SCORE_SCALE
            self.assertAlmostEqual(gap.item(), evaluator(capture) - evaluator(quiet), places=3)
            data = ChessEvalDataset([{'fen': b.fen(), 'score_cp': evaluator(b), 'game_id': i}
                                    for i, b in enumerate([capture, quiet])],
                                   correction_weight=.25, quiet_only=True)
            with torch.inference_mode():
                predicted = data.output_factors * evaluator.model(data.features)
            self.assertTrue(torch.allclose(predicted, data.targets, atol=1e-6))

    def test_leaf_fast_path_preserves_scores_and_skips_terminal_recheck(self):
        from ml.evaluator import NeuralEvaluator
        from engine.search import search
        with tempfile.TemporaryDirectory() as directory:
            path = self.checkpoint(directory)
            fast = NeuralEvaluator(path, .25, quiet_only=True)
            reference = NeuralEvaluator(path, .25, quiet_only=True, optimized=False)
            board = chess.Board()
            for move in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6']:
                board.push_uci(move)
                expected = reference(board)
                with patch('ml.evaluator.evaluate', side_effect=AssertionError('terminal recheck')):
                    self.assertEqual(fast.evaluate_position(board), expected)
            before = board.fen(), list(board.move_stack)
            a = search(board, depth=3, eval_fn=fast)
            b = search(board, depth=3, eval_fn=reference)
            self.assertEqual((a.move, a.score, a.nodes, a.qnodes),
                             (b.move, b.score, b.nodes, b.qnodes))
            self.assertEqual((board.fen(), list(board.move_stack)), before)

    def checkpoint(self, directory):
        import torch
        from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE
        model=ChessNet(color_consistent=True)
        for parameter in model.parameters():
            parameter.data.zero_()
        model.net[-1].bias.data.fill_(1)
        path=Path(directory)/'weighted.pt'
        torch.save({'version':MODEL_VERSION,'input_size':INPUT_SIZE,'score_scale':SCORE_SCALE,
                    'target_mode':'residual','color_consistent':True,'state_dict':model.state_dict()},path)
        return path

    def test_zero_weight_skips_neural_encoding_and_matches_heuristic(self):
        from ml.evaluator import NeuralEvaluator
        from engine.evaluation import evaluate
        with tempfile.TemporaryDirectory() as directory:
            e=NeuralEvaluator(self.checkpoint(directory),0)
            for board in (chess.Board(),chess.Board().mirror()):
                expected=evaluate(board)
                with patch('ml.evaluator.board_to_array',side_effect=AssertionError('should skip')):
                    self.assertEqual(e(board),expected)

    def test_scaled_correction_and_color_symmetry(self):
        from ml.evaluator import NeuralEvaluator
        from engine.evaluation import evaluate
        with tempfile.TemporaryDirectory() as directory:
            path=self.checkpoint(directory)
            for weight in (.1,.25,.5):
                e=NeuralEvaluator(path,weight)
                board=chess.Board()
                self.assertEqual(e(board),evaluate(board)+round(400*weight))
                self.assertEqual(e(board),-e(board.mirror()))

    def test_quiet_gate_skips_captures_and_check(self):
        from ml.evaluator import NeuralEvaluator
        from engine.evaluation import evaluate
        board=chess.Board();board.push_uci('e2e4');board.push_uci('d7d5')
        checking=chess.Board('4k3/8/8/8/8/8/8/4R1K1 b - - 0 1')
        with tempfile.TemporaryDirectory() as directory:
            e=NeuralEvaluator(self.checkpoint(directory),.25,quiet_only=True)
            for b in (board,checking):
                expected=evaluate(b)
                with patch('ml.evaluator.board_to_array',side_effect=AssertionError('should skip')):
                    self.assertEqual(e(b),expected)
            self.assertEqual(e(chess.Board()),evaluate(chess.Board())+100)

    def test_invalid_correction_weights_are_rejected(self):
        from ml.evaluator import NeuralEvaluator
        with tempfile.TemporaryDirectory() as directory:
            path=self.checkpoint(directory)
            for weight in (-.1,1.1,float('nan'),float('inf'),True):
                with self.assertRaises(ValueError):NeuralEvaluator(path,weight)

    def test_settling_follows_capture_promotion_and_check_evasion(self):
        from ml.generate_quiet_rankings import is_tactical
        board=chess.Board();board.push_uci('e2e4');board.push_uci('d7d5')
        self.assertTrue(is_tactical(board,chess.Move.from_uci('e4d5')))
        self.assertFalse(is_tactical(board,chess.Move.from_uci('g1f3')))
        promotion=chess.Board('7k/P7/8/8/8/8/8/K7 w - - 0 1')
        self.assertTrue(is_tactical(promotion,chess.Move.from_uci('a7a8q')))
        checking=chess.Board('4k3/8/8/8/8/8/8/4R1K1 b - - 0 1')
        self.assertTrue(is_tactical(checking,next(iter(checking.legal_moves))))
