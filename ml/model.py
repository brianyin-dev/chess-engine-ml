import torch
import torch.nn as nn


class ChessNet(nn.Module):
    """
    MLP that takes a 768-dim board encoding (12 piece planes × 64 squares)
    and outputs a centipawn evaluation from White's perspective.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def board_to_tensor(board) -> torch.Tensor:
    """
    Encode a chess.Board as a 768-dim float tensor.
    12 planes (6 piece types × 2 colors), each 64 squares, one-hot.
    """
    import chess
    planes = []
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in range(1, 7):  # PAWN=1 .. KING=6
            plane = torch.zeros(64)
            for sq in board.pieces(piece_type, color):
                plane[sq] = 1.0
            planes.append(plane)
    return torch.cat(planes)  # shape: (768,)
