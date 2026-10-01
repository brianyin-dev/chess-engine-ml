"""
Minimal Flask API.

POST /move   { "fen": "<fen>", "depth": 3 }
POST /move   { "moves": ["e2e4", "e7e5"], "depth": 3 }
             -> { "move": "g1f3", "eval": 42 }
"""

from flask import Flask, g, jsonify, request, send_from_directory
import chess
import os
from pathlib import Path
from collections import deque
from threading import BoundedSemaphore, Lock
from time import monotonic

from engine.search import search
from engine.evaluation import evaluate
from engine.opening_book import OpeningBookError, choose_book_move
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_ROOT), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16_384
DEFAULT_BOOK_PATH = PROJECT_ROOT / "books" / "gm2001.bin"
MAX_MOVES = 300
RATE_LIMIT = 120
RATE_WINDOW_SECONDS = 60
search_slots = BoundedSemaphore(1)
rate_lock = Lock()
request_times = {}


@app.errorhandler(413)
def request_too_large(_):
    return jsonify({"error": "request too large", "detail": "Send at most 16 KB of JSON."}), 413


@app.before_request
def limit_move_requests():
    if request.path != "/move" or request.method != "POST":
        return None
    now = monotonic()
    address = request.remote_addr or "unknown"
    with rate_lock:
        # Expire idle clients to bound in-memory rate-limit state.
        for key in list(request_times):
            times = request_times[key]
            while times and times[0] <= now - RATE_WINDOW_SECONDS:
                times.popleft()
            if not times:
                del request_times[key]
        times = request_times.setdefault(address, deque())
        if len(times) >= RATE_LIMIT:
            response = jsonify({"error": "rate limit", "detail": "Too many moves. Try again shortly."})
            response.headers["Retry-After"] = str(max(1, int(RATE_WINDOW_SECONDS - (now - times[0]))))
            return response, 429
        times.append(now)
    if not search_slots.acquire(blocking=False):
        response = jsonify({"error": "engine busy", "detail": "Another game is being analyzed. Try again shortly."})
        response.headers["Retry-After"] = "2"
        return response, 503
    g.search_slot = True
    return None


@app.teardown_request
def release_search_slot(_):
    if getattr(g, "search_slot", False):
        search_slots.release()


@app.get("/")
def index():
    return send_from_directory(FRONTEND_ROOT, "index.html")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


def build_board(payload: dict) -> chess.Board:
    if "moves" in payload:
        if not isinstance(payload["moves"], list) or not all(isinstance(m, str) for m in payload["moves"]):
            raise ValueError("moves must be a list of UCI strings")
        if len(payload["moves"]) > MAX_MOVES:
            raise ValueError(f"moves must contain at most {MAX_MOVES} entries")
        board = chess.Board()
        for uci in payload.get("moves", []):
            if board.is_game_over():
                raise ValueError("move history continues after game over")
            board.push_uci(uci)
        return board

    fen = payload.get("fen", chess.STARTING_FEN)
    if not isinstance(fen, str):
        raise ValueError("fen must be a string")
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("FEN does not describe a valid position")
    return board


def game_status(board: chess.Board) -> dict:
    outcome = board.outcome(claim_draw=False)
    return {"over": outcome is not None,
            "result": outcome.result() if outcome else None,
            "reason": outcome.termination.name.lower() if outcome else None}


@app.route("/move", methods=["POST", "OPTIONS"])
def get_move():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "request must be a JSON object"}), 400

    try:
        depth = data.get("depth", 8)
        time_ms = data.get("time_ms", 750)
        use_book = data.get("use_book", True)
        max_book_ply = data.get("max_book_ply", 20)
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 64:
            raise ValueError("depth must be an integer between 1 and 64")
        if (isinstance(time_ms, bool) or not isinstance(time_ms, (int, float))
                or not math.isfinite(time_ms) or not 1 <= time_ms <= 10_000):
            raise ValueError("time_ms must be a number between 1 and 10000")
        if not isinstance(use_book, bool):
            raise ValueError("use_book must be a boolean")
        if (isinstance(max_book_ply, bool) or not isinstance(max_book_ply, int)
                or not 0 <= max_book_ply <= 100):
            raise ValueError("max_book_ply must be an integer between 0 and 100")
        board = build_board(data)
    except ValueError as exc:
        return jsonify({"error": "invalid request", "detail": str(exc)}), 400

    if board.is_game_over():
        return jsonify({"error": "game over", "game": game_status(board)}), 400

    book_path = Path(os.environ.get("CHESS_BOOK_PATH", DEFAULT_BOOK_PATH))
    book_choice = None
    if use_book and board.ply() < max_book_ply and book_path.is_file():
        try:
            book_choice = choose_book_move(board, book_path)
        except OpeningBookError as exc:
            return jsonify({"error": "opening book error", "detail": str(exc)}), 500

    result = None if book_choice else search(board, depth=depth, time_limit=time_ms / 1000,
                                           use_lmr=True)
    move = book_choice.move if book_choice else result.move
    if move is None:
        return jsonify({"error": "no legal move found"}), 400

    board.push(move)
    response = {
        "move": move.uci(),
        "game": game_status(board),
        "eval": evaluate(board),
        "source": "book" if book_choice else "search",
        "book": ({"file": book_path.name, "weight": book_choice.weight,
                  "alternatives": book_choice.alternatives,
                  "max_ply": max_book_ply} if book_choice else None),
        "search": None if book_choice else {
            "depth": result.depth,
            "nodes": result.nodes,
            "qnodes": result.qnodes,
            "tt_hits": result.tt_hits,
            "static_cache_hits": result.static_cache_hits,
            "fallback_evaluations": result.fallback_evaluations,
            "elapsed_ms": round(result.elapsed * 1000, 2),
            "score": result.score,
            "timed_out": result.timed_out,
            "stop_reason": result.stop_reason,
        },
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1",
            port=int(os.environ.get("PORT", "5000")))
