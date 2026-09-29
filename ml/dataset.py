"""JSONL position labels and pre-encoded tensors for evaluator training."""

import json
from pathlib import Path

import chess
import torch
from torch.utils.data import Dataset

from engine.evaluation import evaluate
from ml.model import SCORE_SCALE, board_to_tensor


def load_rows(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No positions in {path}")
    for row in rows:
        if not isinstance(row.get("game_id"), int) or not isinstance(row.get("fen"), str):
            raise ValueError(f"Invalid position record in {path}")
        if isinstance(row.get("score_cp"), bool) or not isinstance(row.get("score_cp"), int):
            raise ValueError(f"Invalid score in {path}")
    return rows


class ChessEvalDataset(Dataset):
    """Precompute encodings once so training epochs do not repeatedly parse FENs."""

    def __init__(self, rows: list[dict]):
        boards = [chess.Board(row["fen"]) for row in rows]
        self.features = torch.stack([board_to_tensor(board) for board in boards])
        self.baselines = torch.tensor([evaluate(board) for board in boards], dtype=torch.float32)
        self.targets = torch.tensor([(row["score_cp"] - base) / SCORE_SCALE
                                     for row, base in zip(rows, self.baselines.tolist())],
                                    dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]
