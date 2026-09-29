"""Optional adapter from a trained network to the classical search API."""

from pathlib import Path

import chess
import numpy as np
import torch

from engine.evaluation import evaluate
from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE, board_to_array


class NeuralEvaluator:
    cacheable_by_fen = True

    def __init__(self, checkpoint: str | Path):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved.get("version") != MODEL_VERSION or saved.get("input_size") != INPUT_SIZE
                or saved.get("score_scale") != SCORE_SCALE):
            raise ValueError("checkpoint is incompatible with this evaluator")
        self.model = ChessNet()
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)
        linear = (self.model.net[0], self.model.net[2], self.model.net[4])
        self.weights = tuple(layer.weight.detach().numpy() for layer in linear)
        self.biases = tuple(layer.bias.detach().numpy() for layer in linear)

    def __call__(self, board: chess.Board) -> int:
        x = board_to_array(board)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(weight @ x + bias, 0)
        correction = float((self.weights[-1] @ x + self.biases[-1])[0])
        score = evaluate(board) + correction * SCORE_SCALE
        return round(score)
