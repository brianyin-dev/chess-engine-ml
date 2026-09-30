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
from ml.model import ChessNet, INPUT_SIZE, MODEL_VERSION, RELATIONAL_INPUT_SIZE, RELATIONAL_MODEL_VERSION, SCORE_SCALE, board_to_tensor, material_score


def pair_loader(path, shuffle, input_size=INPUT_SIZE, target_mode="material"):
    records = [json.loads(l) for l in path.read_text().splitlines() if l]
    if not records:
        raise ValueError('pair split is empty')
    good = [chess.Board(r['good_fen']) for r in records]
    bad = [chess.Board(r['bad_fen']) for r in records]
    baseline = material_score if target_mode == "material" else evaluate
    tensors = (torch.stack([board_to_tensor(b, input_size) for b in good]),
               torch.stack([board_to_tensor(b, input_size) for b in bad]),
               torch.tensor([(baseline(g) - baseline(b)) / SCORE_SCALE
                             for g, b in zip(good, bad)]),
               torch.tensor([r['sign'] for r in records], dtype=torch.float32))
    importance = torch.tensor([r.get('training_weight', 1) if shuffle else 1 for r in records], dtype=torch.float32)
    if not torch.isfinite(importance).all() or (importance < 1).any():
        raise ValueError('pair training weights must be finite and at least one')
    return DataLoader(torch.utils.data.TensorDataset(*tensors, importance), batch_size=64, shuffle=shuffle)


def ranking_accuracy(model, loader):
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for good, bad, base_difference, sign, _ in loader:
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
              "late_ply_60_plus": [], "low_material_12_pieces_or_fewer": [],
              "search_stand_pat": [], "self_play": []}
    for row, output, baseline, target_base in zip(rows, predictions, dataset.baselines.tolist(),
                                                dataset.target_baselines.tolist()):
        score = output * SCORE_SCALE + target_base
        item = (abs(score - row["score_cp"]), abs(baseline - row["score_cp"]))
        ply = row["ply"]
        key = ("early_ply_8_29" if ply < 30 else
               "middle_ply_30_59" if ply < 60 else "late_ply_60_plus")
        groups[key].append(item)
        groups["search_stand_pat" if 'search' in row.get('source', '')
               else "self_play"].append(item)
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
    parser.add_argument('--pairs', type=Path, help='Candidate pair split directory; material or residual mode')
    parser.add_argument('--rank-weight', type=float, default=.2)
    parser.add_argument('--color-consistent', action='store_true')
    parser.add_argument('--correction-limit-cp', type=float, default=None)
    parser.add_argument('--search-weight', type=float, default=1, help='Training sampling weight for search-derived rows')
    parser.add_argument('--features', choices=('board', 'relationships'), default='board')
    parser.add_argument('--rank-margin-cp', type=float, default=100)
    parser.add_argument('--hard-pair-weight', type=float, default=1)
    parser.add_argument('--selection', choices=('combined', 'ranking'), default='combined')
    parser.add_argument('--initial-checkpoint', type=Path)
    args = parser.parse_args()
    if args.rank_margin_cp <= 0 or args.hard_pair_weight < 1:
        parser.error('positive rank margin and hard-pair weight at least one required')
    if args.selection == 'ranking' and not args.pairs:
        parser.error('ranking selection requires pairs')
    input_size = RELATIONAL_INPUT_SIZE if args.features == 'relationships' else INPUT_SIZE
    version = RELATIONAL_MODEL_VERSION if args.features == 'relationships' else MODEL_VERSION
    if args.search_weight <= 0:
        parser.error('search weight must be positive')
    if args.correction_limit_cp is not None and args.correction_limit_cp <= 0:
        parser.error('correction limit must be positive')
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.patience < 1:
        parser.error("positive epochs, batch-size, learning-rate, and patience required")
    if args.checkpoint.exists() or args.metrics.exists():
        parser.error("choose new checkpoint and metrics paths to preserve prior results")
    if args.pairs and args.target == 'absolute':
        parser.error('ranking requires material or residual target mode')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rows = {split: load_rows(args.data / f"{split}.jsonl") for split in ("train", "val", "test")}
    game_sets = {split: {row["game_id"] for row in data} for split, data in rows.items()}
    if any(game_sets[a] & game_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("game IDs overlap across train/validation/test")
    positions = {split: {row["fen"] for row in data} for split, data in rows.items()}
    if any(positions[a] & positions[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("positions overlap across train/validation/test")
    if args.color_consistent:
        from ml.generate_search_data import key
        canonical = {s: {key(row['fen']) for row in data} for s, data in rows.items()}
        if any(canonical[a] & canonical[b] for a, b in
               (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise ValueError('color-mirrored or equivalent positions overlap across splits')
    datasets = {split: ChessEvalDataset(data, args.target, input_size) for split, data in rows.items()}
    correction_limit = args.correction_limit_cp
    if correction_limit is None and args.target == 'material':
        correction_limit = 250
    model = ChessNet(input_size=input_size, correction_limit_cp=correction_limit, color_consistent=args.color_consistent)
    if args.initial_checkpoint:
        saved = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
        if saved.get('input_size') != input_size or saved.get('target_mode') != args.target or saved.get('diagnostic_only'):
            raise ValueError('initial checkpoint must match features/target and cannot be diagnostic-only')
        model.load_state_dict(saved['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.HuberLoss(delta=1.0)
    weights = [args.search_weight if 'search' in r.get('source', '') else 1
               for r in rows['train']]
    sampler = (torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
               if args.search_weight != 1 else None)
    loader = DataLoader(datasets["train"], batch_size=args.batch_size,
                        shuffle=sampler is None, sampler=sampler)
    pair_loaders = ({s: pair_loader(args.pairs / f'{s}.jsonl', s == 'train', input_size, args.target)
                    for s in ('train', 'val', 'test')} if args.pairs else None)
    if args.pairs:
        pair_records = {s: [json.loads(l) for l in (args.pairs / f'{s}.jsonl').read_text().splitlines()]
                        for s in positions}
        combined_games = {s: game_sets[s] | {r['game_id'] for r in records if 'game_id' in r}
                          for s, records in pair_records.items()}
        if any(combined_games[a] & combined_games[b] for a, b in
               (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise ValueError('candidate-pair game IDs overlap across splits')
        combined = {s: positions[s] | {r[k] for r in
                    [json.loads(l) for l in (args.pairs / f'{s}.jsonl').read_text().splitlines()]
                    for k in ('good_fen', 'bad_fen')} for s in positions}
        if args.color_consistent:
            combined = {s: {key(fen) for fen in fens} for s, fens in combined.items()}
        if any(combined[a] & combined[b] for a, b in
               (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise ValueError('candidate pairs overlap across data splits')
    best_state, best_key, stale = None, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        pairs = cycle(pair_loaders['train']) if pair_loaders else None
        for features, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features), targets)
            if pairs is not None:
                good, bad, base_difference, sign, importance = next(pairs)
                delta = sign * (model(good) - model(bad) + base_difference)
                pair_weights = importance * torch.where(sign * base_difference <= 0, args.hard_pair_weight, 1.)
                rank_loss = torch.relu(args.rank_margin_cp / SCORE_SCALE - delta)
                loss = loss + args.rank_weight * (rank_loss * pair_weights).sum() / pair_weights.sum()
            loss.backward()
            optimizer.step()
        train_mae = mean_absolute_error_cp(model, datasets["train"], args.batch_size)
        val_mae = mean_absolute_error_cp(model, datasets["val"], args.batch_size)
        val_rank = ranking_accuracy(model, pair_loaders['val']) if pair_loaders else None
        selection_score = val_mae + 100 * (1 - val_rank) if val_rank is not None else val_mae
        selection_key = ((1 - val_rank, val_mae) if args.selection == 'ranking' else (selection_score, 0))
        history.append({"epoch": epoch, "train_mae_cp": round(train_mae, 2),
                        "val_mae_cp": round(val_mae, 2), 'val_rank_accuracy': val_rank,
                        'selection_score': selection_score, 'selection_key': selection_key})
        print(f"Epoch {epoch}: train MAE {train_mae:.1f} cp, val MAE {val_mae:.1f} cp", flush=True)
        if best_key is None or selection_key < best_key:
            best_key = selection_key
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": version, "input_size": input_size,
                "score_scale": SCORE_SCALE, "target_mode": args.target,
                'correction_limit_cp': correction_limit,
                'color_consistent': args.color_consistent,
                "state_dict": best_state}, args.checkpoint)
    test_mae = mean_absolute_error_cp(model, datasets["test"], args.batch_size)
    heuristic_mae = mean(abs(evaluate(chess.Board(row["fen"])) - row["score_cp"])
                         for row in rows["test"])
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "seed": args.seed,
        "search_weight": args.search_weight,
        "features": args.features, "input_size": input_size,
        "target_mode": args.target,
        'correction_limit_cp': correction_limit,
        'color_consistent': args.color_consistent,
        "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
        "rows": {split: len(data) for split, data in rows.items()},
        "best_epoch": min(history, key=lambda item: item['selection_key'])["epoch"],
        "validation_mae_cp": round(mean_absolute_error_cp(model, datasets['val'], args.batch_size), 2),
        'selection_policy': ('validation ranking accuracy, then MAE for ties' if args.selection == 'ranking'
                             else 'validation MAE + 100cp times pair error rate' if pair_loaders else 'validation MAE'),
        'rank_margin_cp': args.rank_margin_cp, 'hard_pair_weight': args.hard_pair_weight,
        'initial_checkpoint_sha256': hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest() if args.initial_checkpoint else None,
        'test_rank_accuracy': ranking_accuracy(model, pair_loaders['test']) if pair_loaders else None,
        'rank_weight': args.rank_weight if pair_loaders else 0,
        'weighted_training_pairs': sum(r.get('training_weight', 1) > 1 for r in pair_records['train']) if args.pairs else 0,
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
