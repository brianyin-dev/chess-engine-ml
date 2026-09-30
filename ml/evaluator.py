"""Optional adapter from a trained network to the classical search API."""

from pathlib import Path
import math

import chess
import numpy as np
import torch

from engine.evaluation import evaluate, evaluate_position
from ml.model import (ChessNet, INPUT_SIZE, LEGACY_INPUT_SIZE, MODEL_VERSION,
                      RELATIONAL_INPUT_SIZE, RELATIONAL_MODEL_VERSION,
                      SCORE_SCALE, board_to_array)


class NeuralEvaluator:
    cacheable_by_fen = True

    def __init__(self, checkpoint: str | Path, correction_weight=1.0, quiet_only=False, optimized=True):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ((saved.get("version"), saved.get("input_size")) not in
                ((2, LEGACY_INPUT_SIZE), (MODEL_VERSION, INPUT_SIZE),
                 (RELATIONAL_MODEL_VERSION, RELATIONAL_INPUT_SIZE))
                or saved.get("score_scale") != SCORE_SCALE):
            raise ValueError("checkpoint is incompatible with this evaluator")
        self.target_mode = saved.get("target_mode", "residual")
        if self.target_mode not in ("residual", "absolute", "material"):
            raise ValueError("checkpoint has an unknown target mode")
        if isinstance(correction_weight, bool) or not math.isfinite(correction_weight) or not 0 <= correction_weight <= 1:
            raise ValueError('correction weight must be finite and between zero and one')
        if (correction_weight != 1 or quiet_only) and self.target_mode != 'residual':
            raise ValueError('weighted/gated correction requires a residual checkpoint')
        self.correction_weight = correction_weight
        self.quiet_only = quiet_only
        self.optimized = optimized
        self.diagnostic_only = saved.get('diagnostic_only', False)
        self.input_size = saved['input_size']
        self.correction_limit_cp = saved.get('correction_limit_cp')
        if self.target_mode == 'material' and self.input_size not in (INPUT_SIZE, RELATIONAL_INPUT_SIZE):
            raise ValueError('material mode requires material features')
        if self.correction_limit_cp is not None and self.correction_limit_cp <= 0:
            raise ValueError('correction limit must be positive')
        self.color_consistent = saved.get('color_consistent', False)
        if self.color_consistent and self.input_size not in (INPUT_SIZE, RELATIONAL_INPUT_SIZE):
            raise ValueError('color consistency requires current features')
        self.hidden_sizes = tuple(saved.get('hidden_sizes', (64, 32)))
        self.model = ChessNet(self.input_size, self.correction_limit_cp, self.color_consistent,
                              self.hidden_sizes)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)
        linear = (self.model.net[0], self.model.net[2], self.model.net[4])
        self.weights = tuple(layer.weight.detach().numpy() for layer in linear)
        self.biases = tuple(layer.bias.detach().numpy() for layer in linear)

    def __call__(self, board: chess.Board) -> int:
        baseline = evaluate(board) if self.target_mode == 'residual' else 0
        return self._score(board, baseline)

    def evaluate_position(self, board: chess.Board) -> int:
        """Search-only evaluation after terminal and repetition checks.

        Avoid repeating legal-move generation and replaying repetition history
        at every leaf. Public calls retain the original evaluation semantics.
        """
        if not self.optimized:
            return self(board)
        baseline = evaluate_position(board) if self.target_mode == 'residual' else 0
        return self._score(board, baseline)

    def _score(self, board, baseline):
        if self.target_mode == 'residual' and (self.correction_weight == 0 or
                self.quiet_only and (board.is_check() or next(board.generate_legal_captures(), None))):
            return baseline
        sign = -1 if self.color_consistent and board.turn == chess.BLACK else 1
        encoded_board = board.mirror() if sign == -1 else board
        x = board_to_array(encoded_board, self.input_size)
        material = sign * float(x[792]) * 4000 if self.target_mode == 'material' else 0
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(weight @ x + bias, 0)
        correction = float((self.weights[-1] @ x + self.biases[-1])[0])
        if self.correction_limit_cp is not None:
            limit = self.correction_limit_cp / SCORE_SCALE
            correction = limit * np.tanh(correction / limit)
        score = (baseline if self.target_mode == "residual" else material) + sign * correction * SCORE_SCALE * self.correction_weight
        return round(score)
