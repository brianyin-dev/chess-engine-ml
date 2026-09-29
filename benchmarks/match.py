"""Paired current-versus-legacy matches with equal per-move thinking budgets."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import chess
import chess.pgn
import chess.engine

from benchmarks.legacy.evaluation import evaluate as legacy_evaluate
from benchmarks.legacy.search import best_move as legacy_move
from engine.search import search
from benchmarks.pre_round.search import search as previous_search

ROOT = Path(__file__).resolve().parents[1]


class UciOpponent:
    """Single-threaded UCI opponent, reset between games, with no pondering."""
    def __init__(self, path, elo=None):
        self.path = Path(path).resolve()
        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.path), timeout=10)
        self.options = {"Threads": 1, "Hash": 32}
        if elo is not None:
            limit = self.engine.options.get("UCI_LimitStrength")
            rating = self.engine.options.get("UCI_Elo")
            if limit is None or rating is None:
                self.engine.quit()
                raise ValueError("UCI engine does not support Elo-limited play")
            if rating.min is not None and elo < rating.min or rating.max is not None and elo > rating.max:
                self.engine.quit()
                raise ValueError(f"requested Elo must be between {rating.min} and {rating.max}")
            self.options.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self.engine.configure(self.options)
        self.game = object()

    def new_game(self):
        self.game = object()

    def close(self):
        self.engine.quit()

    def __call__(self, name, board, time_limit, depth_cap):
        if name == "current":
            return choose_move(name, board, time_limit, depth_cap)
        start = perf_counter()
        result = self.engine.play(board, chess.engine.Limit(time=time_limit, depth=depth_cap),
                                  game=self.game, info=chess.engine.INFO_ALL)
        elapsed = perf_counter() - start
        score = result.info.get("score")
        return result.move, {
            "depth": result.info.get("depth", 0), "nodes": result.info.get("nodes"),
            "score": score.pov(board.turn).score(mate_score=100_000) if score else None,
            "elapsed": elapsed, "budget_seconds": time_limit,
            "overrun_seconds": max(0, elapsed - time_limit),
            "timed_out": None, "stop_reason": "uci_returned", "qnodes": None, "tt_hits": None,
        }


class _Deadline(Exception):
    pass


def choose_move(engine, board, time_limit, depth_cap=64):
    """Both engines retain only completed iterations; legacy runs on disposable copies.

    Legacy search/evaluation are unchanged. This adapter adds iterative deepening
    and checks its deadline around leaf evaluation, the extension point it exposes.
    Limits are cooperative, not hard clock forfeits. Actual elapsed time is logged.
    """
    start = perf_counter()
    if engine in ("current", "previous"):
        search_fn = search if engine == "current" else previous_search
        result = search_fn(board, depth=depth_cap, time_limit=time_limit)
        stats = asdict(result)
        move = result.move
        stats.pop("move")
    elif engine == "legacy":
        deadline = start + time_limit
        eval_calls = 0

        def evaluate(position):
            nonlocal eval_calls
            if perf_counter() >= deadline:
                raise _Deadline
            eval_calls += 1
            value = legacy_evaluate(position)
            if perf_counter() >= deadline:
                raise _Deadline
            return value

        move = next(iter(board.legal_moves), None)
        depth = 0
        timed_out = False
        for attempt in range(1, depth_cap + 1):
            if perf_counter() >= deadline:
                timed_out = True
                break
            # The legacy recursion does not pop on exceptions; discard this copy
            # if interrupted, retaining the caller's position and full history.
            working = board.copy(stack=True)
            try:
                candidate = legacy_move(working, depth=attempt, eval_fn=evaluate)
                if perf_counter() >= deadline:
                    raise _Deadline
            except _Deadline:
                timed_out = True
                break
            move, depth = candidate, attempt
        stats = {"depth": depth, "timed_out": timed_out, "eval_calls": eval_calls,
                 "nodes": None, "qnodes": None, "tt_hits": None, "score": None,
                 "stop_reason": "time_limit" if timed_out else None}
    else:
        raise ValueError(f"Unknown engine: {engine}")
    stats["elapsed"] = perf_counter() - start
    stats["budget_seconds"] = time_limit
    stats["overrun_seconds"] = max(0, stats["elapsed"] - time_limit)
    return move, stats


def opening_board(opening):
    if not isinstance(opening, dict) or not isinstance(opening.get("name"), str) or not opening["name"]:
        raise ValueError("each opening needs a nonempty name")
    if not isinstance(opening.get("fen", chess.STARTING_FEN), str):
        raise ValueError("opening fen must be a string")
    board = chess.Board(opening.get("fen", chess.STARTING_FEN))
    if not board.is_valid():
        raise ValueError(f"Invalid position in {opening['name']}")
    moves = opening.get("moves", [])
    if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
        raise ValueError("opening moves must be a list of UCI strings")
    for move in moves:
        if board.is_game_over():
            raise ValueError("opening continues after game over")
        board.push_uci(move)
    if board.is_game_over():
        raise ValueError("opening must end in a playable position")
    return board


def play_game(opening, current_color, time_limit, max_plies, depth_cap=64,
              selector=choose_move, pair_id=1, opponent="legacy"):
    board = opening_board(opening)
    if hasattr(selector, "new_game"):
        selector.new_game()
    game = chess.pgn.Game.from_board(board)
    game.headers.update({
        "Event": "Classical engine comparison", "Site": "Local",
        "Date": datetime.now(timezone.utc).strftime("%Y.%m.%d"),
        "Round": f"{pair_id}.{1 if current_color else 2}",
        "White": "current" if current_color else opponent,
        "Black": opponent if current_color else "current",
        "Opening": opening["name"], "TimeControl": "?",
        "MoveTimeMs": f"{time_limit * 1000:g}",
    })
    node = game.end()
    node.comment = "End of prescribed opening; both engines receive the same per-move budget."
    record = {"pair": pair_id, "opening": opening["name"], "opponent": opponent,
              "initial_fen": board.root().fen(), "opening_moves": [m.uci() for m in board.move_stack],
              "current_color": "white" if current_color else "black", "moves": []}
    error = None
    reason = "ply_limit"
    for _ in range(max_plies):
        if board.is_game_over():
            break
        engine = "current" if board.turn == current_color else opponent
        try:
            # The referee's board cannot be corrupted by an engine adapter.
            move, stats = selector(engine, board.copy(stack=True), time_limit, depth_cap)
            if move not in board.legal_moves:
                raise ValueError(f"{engine} returned an illegal or missing move: {move}")
        except KeyboardInterrupt:
            reason = "interrupted"
            break
        except Exception as exc:
            reason = "engine_error"
            error = {"engine": engine, "type": type(exc).__name__, "message": str(exc)}
            break
        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)
        node.comment = f"{engine}: depth {stats['depth']}, {stats['elapsed']:.4f}s"
        record["moves"].append({"engine": engine, "uci": move.uci(), "san": san, **stats})
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        reason = outcome.termination.name.lower()
        result = outcome.result()
        current_result = "draw" if outcome.winner is None else "win" if outcome.winner == current_color else "loss"
    else:
        result, current_result = "*", None
    record.update({"result": result, "current_result": current_result, "reason": reason,
                   "error": error, "final_fen": board.fen(), "played_plies": len(record["moves"])})
    game.headers["Result"] = result
    game.headers["Termination"] = "normal" if outcome else "unterminated"
    game.headers["TerminationDetail"] = reason
    return record, str(game) + "\n\n"


def summarize(games):
    completed = [g for g in games if g["current_result"] is not None]
    result = {"games": len(games), "completed": len(completed),
              "wins": sum(g["current_result"] == "win" for g in completed),
              "draws": sum(g["current_result"] == "draw" for g in completed),
              "losses": sum(g["current_result"] == "loss" for g in completed),
              "unfinished": sum(g["reason"] == "ply_limit" for g in games),
              "errors": sum(g["reason"] == "engine_error" for g in games),
              "interrupted": sum(g["reason"] == "interrupted" for g in games)}
    result["score_fraction_completed"] = ((result["wins"] + result["draws"] / 2) / len(completed)
                                           if completed else None)
    pairs = {g["pair"] for g in games}
    complete_pairs = [p for p in pairs if sum(g["pair"] == p for g in completed) == 2]
    paired = [g for g in completed if g["pair"] in complete_pairs]
    result["complete_pairs"] = len(complete_pairs)
    result["score_fraction_complete_pairs"] = (
        sum(1 if g["current_result"] == "win" else .5 if g["current_result"] == "draw" else 0
            for g in paired) / len(paired) if paired else None)
    result["timing"] = {}
    for engine in sorted({"current"} | {g.get("opponent", "legacy") for g in games}):
        moves = [m for g in games for m in g["moves"] if m["engine"] == engine]
        result["timing"][engine] = {
            "moves": len(moves),
            "mean_elapsed": sum(m["elapsed"] for m in moves) / len(moves) if moves else None,
            "max_overrun_seconds": max((m["overrun_seconds"] for m in moves), default=0),
            "fallback_moves": sum(m["depth"] == 0 for m in moves),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openings", type=Path, default=ROOT / "benchmarks/openings.json")
    parser.add_argument("--pairs", type=int, help="Use the first N openings; two games per opening")
    parser.add_argument("--time-ms", type=float, default=250)
    parser.add_argument("--max-plies", type=int, default=200, help="Engine-played plies after opening")
    parser.add_argument("--depth-cap", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True, help="New output directory (must not exist)")
    parser.add_argument("--stockfish", type=Path, help="Local Stockfish UCI executable; otherwise play legacy")
    parser.add_argument("--stockfish-elo", type=int,
                        help="Enable the UCI opponent's calibrated limited-strength mode")
    parser.add_argument("--baseline", choices=["legacy", "previous"], default="legacy",
                        help="previous = frozen engine immediately before the final search round")
    args = parser.parse_args()
    if args.stockfish and args.baseline != "legacy":
        parser.error("choose either Stockfish or a local baseline")
    if args.stockfish_elo is not None and not args.stockfish:
        parser.error("stockfish-elo requires --stockfish")
    if not math.isfinite(args.time_ms) or args.time_ms <= 0:
        parser.error("time-ms must be positive and finite")
    if args.max_plies < 1 or not 1 <= args.depth_cap <= 64:
        parser.error("max-plies must be positive; depth-cap must be between 1 and 64")
    try:
        openings = json.loads(args.openings.read_text())
        if not isinstance(openings, list) or not openings:
            raise ValueError("openings must be a nonempty list")
        for opening in openings:
            opening_board(opening)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    pairs = len(openings) if args.pairs is None else args.pairs
    if not 1 <= pairs <= len(openings):
        parser.error("pairs must be between 1 and the number of supplied openings")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("output directory already exists; choose a new directory to preserve prior results")
    sources = ["engine/search.py", "engine/evaluation.py", "benchmarks/legacy/search.py",
               "benchmarks/legacy/evaluation.py", "benchmarks/match.py",
               "benchmarks/pre_round/search.py", "benchmarks/pre_round/evaluation.py"]
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "python_chess": chess.__version__,
        "config": {"time_ms": args.time_ms, "max_plies": args.max_plies,
                   "depth_cap": args.depth_cap, "pairs": pairs, "planned_games": pairs * 2},
        "openings": openings[:pairs],
        "openings_sha256": hashlib.sha256(args.openings.read_bytes()).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "policy": "Equal per-move cooperative deadlines, sequential execution, no pondering. "
                  "Legacy adds iterative deepening and leaf deadline checks to frozen search. "
                  "The previous baseline is the unchanged pre-round timed engine. "
                  "Automatic draws only; no score adjudication or clock forfeits. "
                  "Ply-limit games are unfinished, not draws. Results are not an Elo estimate.",
        "games": [], "status": "running",
    }
    try:
        selector = UciOpponent(args.stockfish, args.stockfish_elo) if args.stockfish else choose_move
    except (OSError, ValueError, chess.engine.EngineError) as exc:
        parser.error(str(exc))
    opponent = (f"stockfish-elo-{args.stockfish_elo}" if args.stockfish_elo is not None
                else "stockfish" if args.stockfish else args.baseline)
    report["config"]["opponent"] = opponent
    if args.stockfish:
        report["opponent"] = {"id": selector.engine.id, "options": selector.options,
                              "path": str(selector.path),
                              "sha256": hashlib.sha256(selector.path.read_bytes()).hexdigest()}
    pgns = []

    def save():
        report["summary"] = summarize(report["games"])
        for name, content in (("report.json", json.dumps(report, indent=2) + "\n"),
                              ("games.pgn", "".join(pgns))):
            temporary = args.output / (name + ".tmp")
            temporary.write_text(content)
            temporary.replace(args.output / name)

    save()
    try:
        for pair_id, opening in enumerate(openings[:pairs], 1):
            # Alternate which engine plays White in the first leg across pairs.
            colors = (chess.WHITE, chess.BLACK) if pair_id % 2 else (chess.BLACK, chess.WHITE)
            for color in colors:
                print(f"Game {len(report['games']) + 1}/{pairs * 2}: {opening['name']}, "
                      f"current as {'White' if color else 'Black'}", flush=True)
                record, pgn = play_game(opening, color, args.time_ms / 1000,
                                        args.max_plies, args.depth_cap, selector=selector,
                                        pair_id=pair_id, opponent=opponent)
                report["games"].append(record)
                pgns.append(pgn)
                save()
                print(f"  {record['result']} ({record['reason']}, {record['played_plies']} plies)", flush=True)
                if record["reason"] == "interrupted":
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    else:
        report["status"] = "completed_with_errors" if summarize(report["games"])["errors"] else "completed"
    finally:
        if args.stockfish:
            selector.close()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Saved {args.output / 'report.json'} and {args.output / 'games.pgn'}")
    if report["status"] == "interrupted":
        raise SystemExit(130)
    if report["status"] == "completed_with_errors":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
