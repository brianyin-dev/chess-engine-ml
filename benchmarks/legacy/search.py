import chess
from benchmarks.legacy.evaluation import evaluate

INF = 10_000_000


def _order_moves(board: chess.Board):
    """Simple move ordering: captures first, then quiet moves."""
    captures = []
    quiets = []
    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            quiets.append(move)
    return captures + quiets


def _alpha_beta(board: chess.Board, depth: int, alpha: int, beta: int, eval_fn) -> int:
    if depth == 0 or board.is_game_over():
        return eval_fn(board)

    for move in _order_moves(board):
        board.push(move)
        score = -_alpha_beta(board, depth - 1, -beta, -alpha, eval_fn)
        board.pop()

        if score >= beta:
            return beta  # beta cutoff
        if score > alpha:
            alpha = score

    return alpha


def best_move(board: chess.Board, depth: int = 3, eval_fn=None) -> chess.Move:
    """
    Returns the best move for the side to move using iterative-deepening
    alpha-beta search.

    eval_fn: callable(chess.Board) -> int, centipawns from White's perspective.
             Defaults to the handcrafted evaluator.
    """
    if eval_fn is None:
        eval_fn = evaluate

    # Negamax expects score from the perspective of the side to move
    def negamax_eval(b: chess.Board) -> int:
        raw = eval_fn(b)
        return raw if b.turn == chess.WHITE else -raw

    best = None
    best_score = -INF

    for move in _order_moves(board):
        board.push(move)
        score = -_alpha_beta(board, depth - 1, -INF, -best_score, negamax_eval)
        board.pop()

        if score > best_score:
            best_score = score
            best = move

    return best
