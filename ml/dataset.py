"""
Dataset utilities.

Expected input: a CSV with columns  fen, eval
  fen  - FEN string of the position
  eval - Stockfish centipawn score (integers; mate scores can be clipped)

Generate this CSV with scripts/generate_data.py (see README).
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.model import board_to_tensor


class ChessEvalDataset(Dataset):
    def __init__(self, csv_path: str, eval_clip: int = 2000):
        df = pd.read_csv(csv_path)
        self.fens = df["fen"].tolist()
        # Clip extreme mate scores so the net isn't overwhelmed
        self.evals = df["eval"].clip(-eval_clip, eval_clip).astype(float).tolist()

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        import chess
        board = chess.Board(self.fens[idx])
        x = board_to_tensor(board)
        y = torch.tensor(self.evals[idx], dtype=torch.float32)
        return x, y
