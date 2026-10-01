"""Time-bounded iterative deepening with quiescence and a per-search cache."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import math
from time import perf_counter

import chess

from engine.evaluation import MATERIAL, MATE_SCORE, evaluate, evaluate_position

INF = 10_000_000
MAX_PLY = 96
MATE_THRESHOLD = MATE_SCORE - MAX_PLY


@dataclass
class SearchResult:
    move: chess.Move | None
    score: int | None  # Centipawns from the root side's perspective; None before depth 1.
    depth: int
    nodes: int
    qnodes: int
    tt_hits: int
    elapsed: float
    timed_out: bool
    stop_reason: str | None = None
    static_cache_hits: int = 0
    fallback_evaluations: int = 0


@dataclass
class _Entry:
    depth: int
    score: int
    bound: str
    move: chess.Move


class _SearchLimit(Exception):
    pass


def _position_key(board):
    # Include every rule-relevant part of a position; no private python-chess API.
    return (
        board.pawns, board.knights, board.bishops, board.rooks, board.queens,
        board.kings, board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
        board.turn, board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def _to_tt(score, ply):
    return score + ply if score >= MATE_THRESHOLD else score - ply if score <= -MATE_THRESHOLD else score


def _from_tt(score, ply):
    return score - ply if score >= MATE_THRESHOLD else score + ply if score <= -MATE_THRESHOLD else score


def _tactical_moves(board):
    """All legal captures and promotions, without generating other quiet moves."""
    moves = list(board.generate_legal_captures())
    penultimate = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
    for move in board.generate_legal_moves(from_mask=board.pawns & penultimate,
                                           to_mask=~board.occupied & chess.BB_ALL):
        if move.promotion:
            moves.append(move)
    return moves


class _Search:
    def __init__(self, board, eval_fn, deadline, use_tt, node_limit=None):
        self.eval_fn = eval_fn
        self.deadline = deadline
        self.node_limit = node_limit
        self.use_tt = use_tt
        self.table = {}
        self.static_cache = {}
        self.static_cache_hits = 0
        self.fallback_evaluations = 0
        self.move_hints = {}
        self.killers = {}
        self.history = {}
        self.nodes = self.qnodes = self.tt_hits = 0
        self.root_candidate = None
        self.use_pvs = True
        # FEN alone cannot reconstruct repetitions. Preserve supplied move history.
        replay = board.copy(stack=True)
        self.counts = Counter({_position_key(replay): 1})
        for _ in range(min(replay.halfmove_clock, len(replay.move_stack))):
            replay.pop()
            self.counts[_position_key(replay)] += 1

    def cache_key(self, board):
        # A position-only cache can reuse a score from a different repetition or
        # 75-move context. Retain the reversible history counts for correctness.
        return (_position_key(board), board.halfmove_clock, frozenset(self.counts.items()))

    @contextmanager
    def pushed(self, board, move):
        old_counts = self.counts
        zeroing = board.is_zeroing(move)
        board.push(move)
        key = _position_key(board)
        if zeroing:
            self.counts = Counter()
        self.counts[key] += 1
        try:
            yield
        finally:
            self.counts[key] -= 1
            if not self.counts[key]:
                del self.counts[key]
            self.counts = old_counts
            board.pop()

    def visit(self, ply, quiescence=False):
        if self.node_limit is not None and self.nodes >= self.node_limit:
            raise _SearchLimit("node_limit")
        if self.deadline is not None and perf_counter() >= self.deadline:
            raise _SearchLimit("time_limit")
        # Abort the iteration rather than treating an unresolved checking line as quiet.
        if ply >= MAX_PLY:
            raise _SearchLimit("ply_limit")
        self.nodes += 1
        self.qnodes += int(quiescence)

    def terminal(self, board, moves, ply):
        if not moves:
            return -MATE_SCORE + ply if board.is_check() else 0
        if (board.is_insufficient_material() or board.halfmove_clock >= 150
                or self.counts[_position_key(board)] >= 5):
            return 0
        # Claims (threefold/50 moves) require a claim action, which this app does
        # not expose. Match python-chess's default automatic-draw policy.
        return None

    def static(self, board):
        # Only evaluators explicitly known to depend on position state alone are
        # cached. Terminal/repetition rules have already been checked.
        if self.eval_fn is evaluate:
            key = _position_key(board)
            if key in self.static_cache:
                self.static_cache_hits += 1
                score = self.static_cache[key]
            else:
                score = evaluate_position(board)
                if len(self.static_cache) >= 20_000:
                    self.static_cache.clear()
                self.static_cache[key] = score
        elif getattr(self.eval_fn, "cacheable_by_fen", False):
            key = (_position_key(board), min(board.halfmove_clock, 150))
            if key in self.static_cache:
                self.static_cache_hits += 1
                score = self.static_cache[key]
            else:
                position_evaluator = getattr(self.eval_fn, 'evaluate_position', self.eval_fn)
                score = int(position_evaluator(board))
                if len(self.static_cache) >= 20_000:
                    self.static_cache.clear()
                self.static_cache[key] = score
        else:
            score = int(self.eval_fn(board))
        # Terminal scores belong to the search even when using a learned evaluator.
        score = max(-MATE_THRESHOLD + 1, min(MATE_THRESHOLD - 1, score))
        return score if board.turn == chess.WHITE else -score

    def ordered(self, board, moves, preferred=None, ply=0):
        def priority(move):
            if move == preferred:
                return 10_000_000
            score = 0
            if board.is_capture(move):
                victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
                score += 100_000 + 10 * MATERIAL[victim] - MATERIAL[board.piece_type_at(move.from_square)]
            if move.promotion:
                score += 100_000 + MATERIAL[move.promotion]
            if not score and move in self.killers.get(ply, ()):
                score += 60_000
            return score + self.history.get((board.turn, move), 0)
        return sorted(moves, key=priority, reverse=True)

    def quiescence(self, board, alpha, beta, ply):
        self.visit(ply, quiescence=True)
        in_check = board.is_check()
        # A quiet node needs just one legal move to rule out stalemate. Avoid
        # generating and sorting every quiet move at every quiescence leaf.
        moves = list(board.legal_moves) if in_check else None
        has_moves = moves if in_check else next(iter(board.legal_moves), None)
        terminal = self.terminal(board, has_moves, ply)
        if terminal is not None:
            return terminal
        if not in_check:
            stand_pat = self.static(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            moves = _tactical_moves(board)
        # In check, standing still is illegal: search every legal evasion.
        for move in self.ordered(board, moves, ply=ply):
            with self.pushed(board, move):
                score = -self.quiescence(board, -beta, -alpha, ply + 1)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)
        self.visit(ply)
        moves = list(board.legal_moves)
        terminal = self.terminal(board, moves, ply)
        if terminal is not None:
            return terminal
        original_alpha = alpha
        key = self.cache_key(board) if self.use_tt else None
        entry = self.table.get(key)
        if entry and entry.depth >= depth:
            score = _from_tt(entry.score, ply)
            if (entry.bound == "exact" or entry.bound == "lower" and score >= beta
                    or entry.bound == "upper" and score <= alpha):
                self.tt_hits += 1
                if ply == 0:
                    self.root_candidate = entry.move
                return score
        best_score, best = -INF, None
        preferred = entry.move if entry else self.move_hints.get(_position_key(board))
        for index, move in enumerate(self.ordered(board, moves, preferred, ply=ply)):
            quiet = not board.is_capture(move) and not move.promotion
            with self.pushed(board, move):
                if self.use_pvs and index > 0 and beta > alpha + 1:
                    # Probe later moves cheaply. A move that improves alpha
                    # inside the window needs a full search before accepting it.
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            if score > best_score:
                best_score, best = score, move
                if ply == 0:
                    self.root_candidate = move
            alpha = max(alpha, score)
            if alpha >= beta:
                if quiet:
                    killers = self.killers.setdefault(ply, [])
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]
                    history_key = (board.turn, move)
                    self.history[history_key] = min(50_000, self.history.get(history_key, 0) + depth * depth)
                break
        # Position-only hints change ordering, never reuse history-dependent scores.
        if len(self.move_hints) >= 20_000:
            self.move_hints.clear()
        self.move_hints[_position_key(board)] = best
        if self.use_tt:
            bound = "upper" if best_score <= original_alpha else "lower" if best_score >= beta else "exact"
            if len(self.table) >= 20_000:
                self.table.clear()
            self.table[key] = _Entry(depth, _to_tt(best_score, ply), bound, best)
        return best_score

    def fallback(self, board, moves, deadline):
        """Cheap positional pass with a conservative moved-piece safety penalty.

        This is an emergency preference, not a completed minimax iteration. It
        uses only a small part of the existing budget and retains legal fallback
        behavior if even that budget has already elapsed.
        """
        best, best_score = moves[0], -INF
        color = board.turn
        for move in self.ordered(board, moves):
            if deadline is not None and perf_counter() >= deadline:
                break
            with self.pushed(board, move):
                terminal = self.terminal(board, next(iter(board.legal_moves), None), 1)
                if terminal is not None:
                    score = -terminal
                else:
                    score = evaluate_position(board) * (1 if color else -1)
                    attackers = board.attackers(not color, move.to_square)
                    if attackers and board.piece_type_at(move.to_square) != chess.KING:
                        value = MATERIAL[board.piece_type_at(move.to_square)]
                        defended = bool(board.attackers(color, move.to_square))
                        cheapest = min(20_000 if board.piece_type_at(sq) == chess.KING
                                       else MATERIAL[board.piece_type_at(sq)] for sq in attackers)
                        score -= max(0, value - cheapest) if defended else value
            self.fallback_evaluations += 1
            if score > best_score:
                best, best_score = move, score
        return best


def search(board: chess.Board, depth: int = 3, eval_fn=None,
           time_limit: float | None = None, use_tt: bool = True,
           node_limit: int | None = None) -> SearchResult:
    """Search up to depth plies; return the last fully completed iteration.

    eval_fn returns integer centipawns from White's perspective. time_limit is
    seconds, checked cooperatively between nodes. Even a tiny limit returns a
    legal fallback move. node_limit counts main and quiescence visits; fallback
    static evaluations are separately recorded and are not visited nodes.
    The caller's board is restored, including on exceptions.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 64:
        raise ValueError("depth must be an integer between 1 and 64")
    if time_limit is not None and (isinstance(time_limit, bool)
            or not isinstance(time_limit, (int, float))
            or not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be a positive finite number of seconds")
    if node_limit is not None and (isinstance(node_limit, bool)
            or not isinstance(node_limit, int) or node_limit < 1):
        raise ValueError("node_limit must be a positive integer")
    if not board.is_valid():
        raise ValueError("invalid chess position")
    start = perf_counter()
    worker = _Search(board, evaluate if eval_fn is None else eval_fn,
                     None if time_limit is None else start + time_limit, use_tt, node_limit)
    moves = list(board.legal_moves)
    terminal = worker.terminal(board, moves, 0)
    if terminal is not None:
        return SearchResult(None, terminal, 0, 0, 0, 0, perf_counter() - start, False)
    fallback_deadline = None if time_limit is None else min(start + time_limit, perf_counter() + time_limit * .1)
    best = worker.fallback(board, moves, fallback_deadline)
    worker.move_hints[_position_key(board)] = best
    score, completed, stop_reason = None, 0, None
    for current_depth in range(1, depth + 1):
        try:
            candidate_score = worker.negamax(board, current_depth, -INF, INF, 0)
        except _SearchLimit as exc:
            stop_reason = str(exc)
            break
        best, score, completed = worker.root_candidate, candidate_score, current_depth
        if abs(score) >= MATE_THRESHOLD:
            break
    return SearchResult(best, score, completed, worker.nodes, worker.qnodes,
                        worker.tt_hits, perf_counter() - start,
                        stop_reason == "time_limit", stop_reason,
                        worker.static_cache_hits, worker.fallback_evaluations)


def best_move(board: chess.Board, depth: int = 3, eval_fn=None,
              time_limit: float | None = None) -> chess.Move | None:
    """Compatibility wrapper; use search() for scores and diagnostics."""
    return search(board, depth=depth, eval_fn=eval_fn, time_limit=time_limit).move
