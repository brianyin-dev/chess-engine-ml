"""Small position evaluator trained on White-perspective Stockfish scores."""

import chess
import numpy as np
import torch
from torch import nn


INPUT_SIZE = 782  # 12 piece planes, turn, four castling flags, EP file, clock.
SCORE_SCALE = 400.0  # Network targets are centipawns divided by this value.
MODEL_VERSION = 2


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def board_to_array(board: chess.Board) -> np.ndarray:
    """Encode pieces and rule-relevant state without consulting move history."""
    values = np.zeros(INPUT_SIZE, dtype=np.float32)
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
    return values


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    return torch.from_numpy(board_to_array(board))
