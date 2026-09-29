"""Small position evaluator trained on White-perspective Stockfish scores."""

import chess
import numpy as np
import torch
from torch import nn
from engine.evaluation import MATERIAL, PHASE_WEIGHTS


LEGACY_INPUT_SIZE = 782
INPUT_SIZE = 794  # Legacy features plus ten material counts, balance, phase.
SCORE_SCALE = 400.0  # Network targets are centipawns divided by this value.
MODEL_VERSION = 3


class ChessNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, correction_limit_cp=None):
        super().__init__()
        self.correction_limit_cp = correction_limit_cp
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x).squeeze(-1)
        if self.correction_limit_cp is not None:
            limit = self.correction_limit_cp / SCORE_SCALE
            score = limit * torch.tanh(score / limit)
        return score


def material_score(board: chess.Board) -> int:
    return sum(MATERIAL[p] * (board.pieces_mask(p, chess.WHITE).bit_count() -
                            board.pieces_mask(p, chess.BLACK).bit_count())
               for p in range(1, 6))


def board_to_array(board: chess.Board, input_size=INPUT_SIZE) -> np.ndarray:
    """Encode pieces and rule-relevant state without consulting move history."""
    if input_size not in (LEGACY_INPUT_SIZE, INPUT_SIZE):
        raise ValueError('unsupported feature size')
    values = np.zeros(input_size, dtype=np.float32)
    for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
        for piece_type in chess.PIECE_TYPES:
            offset = (color_index * 6 + piece_type - 1) * 64
            for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                values[offset + square] = 1.0
    values[768] = float(board.turn == chess.WHITE)
    for index, enabled in enumerate((
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )):
        values[769 + index] = float(enabled)
    if board.ep_square is not None and board.has_legal_en_passant():
        values[773 + chess.square_file(board.ep_square)] = 1.0
    values[781] = min(board.halfmove_clock, 150) / 150.0
    if input_size == INPUT_SIZE:
        phase = 0
        for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
            for piece_type in range(1, 6):
                count = board.pieces_mask(piece_type, color).bit_count()
                values[782 + color_index * 5 + piece_type - 1] = count / (8 if piece_type == 1 else 2)
                phase += count * PHASE_WEIGHTS.get(piece_type, 0)
        values[792] = material_score(board) / 4000.0
        values[793] = min(phase, 24) / 24.0
    return values


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    return torch.from_numpy(board_to_array(board))
