"""Train a compact evaluator and compare held-out error with the heuristic."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from itertools import cycle

import chess
import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.evaluation import evaluate
from ml.dataset import ChessEvalDataset, load_rows
from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, SCORE_SCALE, board_to_tensor, material_score


def pair_loader(path, shuffle):
    records = [json.loads(l) for l in path.read_text().splitlines() if l]
    if not records:
        raise ValueError('pair split is empty')
    good = [chess.Board(r['good_fen']) for r in records]
    bad = [chess.Board(r['bad_fen']) for r in records]
    tensors = (torch.stack([board_to_tensor(b) for b in good]),
               torch.stack([board_to_tensor(b) for b in bad]),
               torch.tensor([(material_score(g) - material_score(b)) / SCORE_SCALE
                             for g, b in zip(good, bad)]),
               torch.tensor([r['sign'] for r in records], dtype=torch.float32))
    return DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=64, shuffle=shuffle)


def ranking_accuracy(model, loader):
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for good, bad, base_difference, sign in loader:
            delta = sign * (model(good) - model(bad) + base_difference)
            correct += (delta > 0).sum().item()
            total += len(delta)
    return correct / total


def mean_absolute_error_cp(model, dataset, batch_size: int) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for features, targets in DataLoader(dataset, batch_size=batch_size):
            total += (model(features) - targets).abs().sum().item() * SCORE_SCALE
    return total / len(dataset)


def heldout_breakdown(model, dataset, rows, batch_size: int, target_mode: str) -> dict:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for features, _ in DataLoader(dataset, batch_size=batch_size):
            predictions.extend(model(features).tolist())
    groups = {"early_ply_8_29": [], "middle_ply_30_59": [],
              "late_ply_60_plus": [], "low_material_12_pieces_or_fewer": []}
    for row, output, baseline, target_base in zip(rows, predictions, dataset.baselines.tolist(),
                                                dataset.target_baselines.tolist()):
        score = output * SCORE_SCALE + target_base
        item = (abs(score - row["score_cp"]), abs(baseline - row["score_cp"]))
        ply = row["ply"]
        key = ("early_ply_8_29" if ply < 30 else
               "middle_ply_30_59" if ply < 60 else "late_ply_60_plus")
        groups[key].append(item)
        if len(chess.Board(row["fen"]).piece_map()) <= 12:
            groups["low_material_12_pieces_or_fewer"].append(item)
    return {name: {"positions": len(items),
                   "network_mae_cp": round(mean(x[0] for x in items), 2) if items else None,
                   "heuristic_mae_cp": round(mean(x[1] for x in items), 2) if items else None}
            for name, items in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=("residual", "absolute", "material"), default="residual")
    parser.add_argument('--pairs', type=Path, help='Candidate pair split directory; material mode only')
    parser.add_argument('--rank-weight', type=float, default=.2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.patience < 1:
        parser.error("positive epochs, batch-size, learning-rate, and patience required")
    if args.checkpoint.exists() or args.metrics.exists():
        parser.error("choose new checkpoint and metrics paths to preserve prior results")
    if args.pairs and args.target != 'material':
        parser.error('ranking currently requires material target mode')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rows = {split: load_rows(args.data / f"{split}.jsonl") for split in ("train", "val", "test")}
    game_sets = {split: {row["game_id"] for row in data} for split, data in rows.items()}
    if any(game_sets[a] & game_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("game IDs overlap across train/validation/test")
    positions = {split: {row["fen"] for row in data} for split, data in rows.items()}
    if any(positions[a] & positions[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("positions overlap across train/validation/test")
    datasets = {split: ChessEvalDataset(data, args.target) for split, data in rows.items()}
    correction_limit = 250 if args.target == 'material' else None
    model = ChessNet(correction_limit_cp=correction_limit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.HuberLoss(delta=1.0)
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    pair_loaders = ({s: pair_loader(args.pairs / f'{s}.jsonl', s == 'train')
                    for s in ('train', 'val', 'test')} if args.pairs else None)
    if args.pairs:
        combined = {s: positions[s] | {r[k] for r in
                    [json.loads(l) for l in (args.pairs / f'{s}.jsonl').read_text().splitlines()]
                    for k in ('good_fen', 'bad_fen')} for s in positions}
        if any(combined[a] & combined[b] for a, b in
               (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise ValueError('candidate pairs overlap across data splits')
    best_state, best_val, stale = None, float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        pairs = cycle(pair_loaders['train']) if pair_loaders else None
        for features, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features), targets)
            if pairs is not None:
                good, bad, base_difference, sign = next(pairs)
                delta = sign * (model(good) - model(bad) + base_difference)
                loss = loss + args.rank_weight * torch.relu(.25 - delta).mean()
            loss.backward()
            optimizer.step()
        train_mae = mean_absolute_error_cp(model, datasets["train"], args.batch_size)
        val_mae = mean_absolute_error_cp(model, datasets["val"], args.batch_size)
        val_rank = ranking_accuracy(model, pair_loaders['val']) if pair_loaders else None
        selection_score = val_mae + 100 * (1 - val_rank) if val_rank is not None else val_mae
        history.append({"epoch": epoch, "train_mae_cp": round(train_mae, 2),
                        "val_mae_cp": round(val_mae, 2), 'val_rank_accuracy': val_rank,
                        'selection_score': selection_score})
        print(f"Epoch {epoch}: train MAE {train_mae:.1f} cp, val MAE {val_mae:.1f} cp", flush=True)
        if selection_score < best_val:
            best_val = selection_score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": MODEL_VERSION, "input_size": INPUT_SIZE,
                "score_scale": SCORE_SCALE, "target_mode": args.target,
                'correction_limit_cp': correction_limit,
                "state_dict": best_state}, args.checkpoint)
    test_mae = mean_absolute_error_cp(model, datasets["test"], args.batch_size)
    heuristic_mae = mean(abs(evaluate(chess.Board(row["fen"])) - row["score_cp"])
                         for row in rows["test"])
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "seed": args.seed,
        "target_mode": args.target,
        'correction_limit_cp': correction_limit,
        "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
        "rows": {split: len(data) for split, data in rows.items()},
        "best_epoch": min(history, key=lambda item: item['selection_score'])["epoch"],
        "validation_mae_cp": round(mean_absolute_error_cp(model, datasets['val'], args.batch_size), 2),
        'selection_policy': 'validation MAE + 100cp times pair error rate' if pair_loaders else 'validation MAE',
        'test_rank_accuracy': ranking_accuracy(model, pair_loaders['test']) if pair_loaders else None,
        'rank_weight': args.rank_weight if pair_loaders else 0,
        'pair_manifest_sha256': hashlib.sha256((args.pairs / 'manifest.json').read_bytes()).hexdigest()
                                if args.pairs else None,
        "test_mae_cp": round(test_mae, 2),
        "heuristic_test_mae_cp": round(heuristic_mae, 2),
        "test_breakdown": heldout_breakdown(model, datasets["test"], rows["test"],
                                             args.batch_size, args.target),
        "history": history,
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Held-out test MAE: NN {test_mae:.1f} cp; heuristic {heuristic_mae:.1f} cp")


if __name__ == "__main__":
    main()
