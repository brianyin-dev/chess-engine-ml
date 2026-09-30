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
RELATIONAL_INPUT_SIZE = INPUT_SIZE + 98
RELATIONAL_MODEL_VERSION = 4


class ChessNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, correction_limit_cp=None, color_consistent=False):
        super().__init__()
        self.correction_limit_cp = correction_limit_cp
        if color_consistent and input_size not in (INPUT_SIZE, RELATIONAL_INPUT_SIZE):
            raise ValueError("color consistency requires current features")
        self.color_consistent = color_consistent
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sign = 1
        if self.color_consistent:
            # Canonicalize to the mover as White; restore White-perspective sign.
            mirrored = x.clone()
            planes = x[..., :768].reshape(*x.shape[:-1], 2, 6, 8, 8)
            mirrored[..., :768] = planes.flip((-4, -2)).reshape(*x.shape[:-1], 768)
            mirrored[..., 768] = 1 - x[..., 768]
            mirrored[..., 769:773] = x[..., [771, 772, 769, 770]]
            mirrored[..., 782:792] = x[..., [787, 788, 789, 790, 791, 782, 783, 784, 785, 786]]
            mirrored[..., 792] = -x[..., 792]
            if x.shape[-1] == RELATIONAL_INPUT_SIZE:
                mirrored[..., 794:] = x[..., 794:].reshape(*x.shape[:-1], 2, 49).flip(-2).reshape(*x.shape[:-1], 98)
            white = x[..., 768] > .5
            sign = torch.where(white, 1., -1.)
            x = torch.where(white.unsqueeze(-1), x, mirrored)
        score = self.net(x).squeeze(-1)
        if self.correction_limit_cp is not None:
            limit = self.correction_limit_cp / SCORE_SCALE
            score = limit * torch.tanh(score / limit)
        return score * sign


def material_score(board: chess.Board) -> int:
    return sum(MATERIAL[p] * (board.pieces_mask(p, chess.WHITE).bit_count() -
                            board.pieces_mask(p, chess.BLACK).bit_count())
               for p in range(1, 6))


def board_to_array(board: chess.Board, input_size=INPUT_SIZE) -> np.ndarray:
    """Encode pieces and rule-relevant state without consulting move history."""
    if input_size not in (LEGACY_INPUT_SIZE, INPUT_SIZE, RELATIONAL_INPUT_SIZE):
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
    if input_size in (INPUT_SIZE, RELATIONAL_INPUT_SIZE):
        phase = 0
        for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
            for piece_type in range(1, 6):
                count = board.pieces_mask(piece_type, color).bit_count()
                values[782 + color_index * 5 + piece_type - 1] = count / (8 if piece_type == 1 else 2)
                phase += count * PHASE_WEIGHTS.get(piece_type, 0)
        values[792] = material_score(board) / 4000.0
        values[793] = min(phase, 24) / 24.0
    if input_size == RELATIONAL_INPUT_SIZE:
        values[794:] = relationship_features(board)
    return values


def board_to_tensor(board: chess.Board, input_size=INPUT_SIZE) -> torch.Tensor:
    return torch.from_numpy(board_to_array(board, input_size))


# Reuse immutable king/square geometry instead of recomputing distances per leaf.
_KING_GEOMETRY = tuple(tuple((
    1 / (1 + max(abs((sq >> 3) - (king >> 3)), abs((sq & 7) - (king & 7)))),
    ((sq >> 3) - (king >> 3)) / 7,
    abs((sq & 7) - (king & 7)) / 7)
    for sq in range(64)) for king in range(64))


def relationship_features(board):
    """49 normalized, mirror-equivariant relationship features per color.

    Per piece type: own/enemy king proximity, forward offset, lateral offset;
    attacked, defended, attacked-and-undefended counts. Four pawn features follow.
    Attack maps include pinned attackers, as python-chess pseudo-attacks do.
    """
    attacks = {}
    for color in (chess.WHITE, chess.BLACK):
        mask = 0
        for sq in chess.scan_forward(board.occupied_co[color]):
            mask |= board.attacks_mask(sq)
        attacks[color] = mask
    result = []
    for color in (chess.WHITE, chess.BLACK):
        direction = 1 if color else -1
        own_king, enemy_king = board.king(color), board.king(not color)
        pieces = [list(chess.scan_forward(board.pieces_mask(pt, color))) for pt in range(1, 6)]
        for squares in pieces:
            for king in (own_king, enemy_king):
                proximity = forward = lateral = 0.
                if king is not None:
                    geometry = _KING_GEOMETRY[king]
                    for sq in squares:
                        a, b, c = geometry[sq]
                        proximity += a
                        forward += b
                        lateral += c
                result.extend((proximity / 8, direction * forward / 8, lateral / 8))
        for pt in range(1, 6):
            occupied = board.pieces_mask(pt, color)
            attacked = occupied & attacks[not color]
            defended = occupied & attacks[color]
            hanging = attacked & ~defended
            result.extend((attacked.bit_count() / 8, defended.bit_count() / 8, hanging.bit_count() / 8))
        pawns = pieces[0]
        enemy = list(chess.scan_forward(board.pieces_mask(chess.PAWN, not color)))
        files = [sq & 7 for sq in pawns]
        distinct = set(files)
        isolated = sum((f - 1) not in distinct and (f + 1) not in distinct for f in files)
        doubled = len(files) - len(distinct)
        passed = sum(not any(abs((sq & 7) - (e & 7)) <= 1 and
                            direction * ((e >> 3) - (sq >> 3)) > 0
                            for e in enemy) for sq in pawns)
        shield = sum(own_king is not None and abs((sq & 7) - (own_king & 7)) <= 1 and
                     0 < direction * ((sq >> 3) - (own_king >> 3)) <= 2
                     for sq in pawns)
        result.extend((isolated / 8, doubled / 8, passed / 8, shield / 8))
    return np.asarray(result, dtype=np.float32)
