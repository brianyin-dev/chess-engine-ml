import chess

# Material values in centipawns
MATERIAL = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

MOBILITY_WEIGHTS = {
    chess.KNIGHT: 4,
    chess.BISHOP: 5,
    chess.ROOK: 2,
    chess.QUEEN: 1,
}

DEVELOPMENT_BONUS = 14
CENTER_CONTROL_BONUS = 10
ROOK_OPEN_FILE_BONUS = 18
ROOK_SEMI_OPEN_FILE_BONUS = 10
PASSED_PAWN_BONUS = [0, 8, 14, 24, 40, 62, 100, 0]
QUEEN_EARLY_DEVELOPMENT_PENALTY = 10
MATE_SCORE = 100_000
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
MAX_PHASE = 24

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
PASSED_MASKS = {
    color: [sum(
        chess.BB_SQUARES[chess.square(file, rank)]
        for file in range(max(0, chess.square_file(square) - 1), min(8, chess.square_file(square) + 2))
        for rank in (range(chess.square_rank(square) + 1, 8) if color == chess.WHITE
                     else range(chess.square_rank(square)))
    ) for square in chess.SQUARES]
    for color in chess.COLORS
}

# Tables are written visually from White's perspective: a8 first, h1 last.
# fmt: off
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}
# fmt: on


def _phase(board: chess.Board) -> int:
    """24 at the start, approaching zero as non-pawn material disappears."""
    return min(MAX_PHASE, sum(
        weight * chess.popcount(board.pieces_mask(piece_type, color))
        for piece_type, weight in PHASE_WEIGHTS.items()
        for color in chess.COLORS
    ))


def _pst_score(board: chess.Board, color: chess.Color, phase=None) -> int:
    score = 0
    if phase is None:
        phase = _phase(board)
    for piece_type, table in PST.items():
        for sq in chess.scan_forward(board.pieces_mask(piece_type, color)):
            idx = chess.square_mirror(sq) if color == chess.WHITE else sq
            value = table[idx]
            if piece_type == chess.KING:
                # Encourage a sheltered king early and an active king in endings.
                file, rank = chess.square_file(sq), chess.square_rank(sq)
                center_distance = abs(2 * file - 7) + abs(2 * rank - 7)
                endgame = 40 - 10 * center_distance
                value = (value * phase + endgame * (MAX_PHASE - phase)) // MAX_PHASE
            score += value
    return score


def _material_score(board: chess.Board, color: chess.Color) -> int:
    return sum(MATERIAL[pt] * chess.popcount(board.pieces_mask(pt, color)) for pt in MATERIAL)


def _mobility_score(board: chess.Board, color: chess.Color) -> int:
    """Pseudo-mobility: attacked squares excluding friendly-occupied squares.

    This is a positional feature, not legal move generation. It works for either
    color without mutating turn, castling, en-passant, or repetition state.
    """
    score = 0
    friendly = board.occupied_co[color]
    for piece_type, weight in MOBILITY_WEIGHTS.items():
        for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
            score += weight * chess.popcount(board.attacks_mask(square) & ~friendly)
    return score


def _development_score(board: chess.Board, color: chess.Color, phase: int) -> int:
    home_rank = 0 if color == chess.WHITE else 7
    score = 0

    minors = board.pieces_mask(chess.KNIGHT, color) | board.pieces_mask(chess.BISHOP, color)
    for square in chess.scan_forward(minors):
        if chess.square_rank(square) != home_rank:
            score += DEVELOPMENT_BONUS

    queen_home = chess.D1 if color == chess.WHITE else chess.D8
    queens = board.pieces_mask(chess.QUEEN, color)
    if queens and not queens & chess.BB_SQUARES[queen_home]:
        undeveloped_minors = 0
        for square in chess.scan_forward(minors):
            if chess.square_rank(square) == home_rank:
                undeveloped_minors += 1
        if undeveloped_minors >= 2:
            score -= QUEEN_EARLY_DEVELOPMENT_PENALTY

    return score * phase // MAX_PHASE


def _center_control_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for square in CENTER_SQUARES:
        if board.is_attacked_by(color, square):
            score += CENTER_CONTROL_BONUS
    return score


def _rook_activity_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    enemy = not color
    for square in chess.scan_forward(board.pieces_mask(chess.ROOK, color)):
        file_index = chess.square_file(square)
        friendly_pawns = board.pieces_mask(chess.PAWN, color) & chess.BB_FILES[file_index]
        enemy_pawns = board.pieces_mask(chess.PAWN, enemy) & chess.BB_FILES[file_index]
        if not friendly_pawns and not enemy_pawns:
            score += ROOK_OPEN_FILE_BONUS
        elif not friendly_pawns:
            score += ROOK_SEMI_OPEN_FILE_BONUS
    return score


def _passed_pawn_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    enemy_pawns = board.pieces_mask(chess.PAWN, not color)
    for square in chess.scan_forward(board.pieces_mask(chess.PAWN, color)):
        rank_index = chess.square_rank(square)
        if not enemy_pawns & PASSED_MASKS[color][square]:
            progress = rank_index if color == chess.WHITE else 7 - rank_index
            score += PASSED_PAWN_BONUS[progress]
    return score


def _activity_score(board: chess.Board, color: chess.Color, phase: int) -> int:
    return (
        _development_score(board, color, phase)
        + _center_control_score(board, color)
        + _rook_activity_score(board, color)
        + _passed_pawn_score(board, color)
    )


def evaluate(board: chess.Board) -> int:
    """
    Returns a centipawn score from White's perspective.
    Positive = White is better, negative = Black is better.
    """
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
        return 0

    return evaluate_position(board)


def evaluate_position(board: chess.Board) -> int:
    """Positional score only; search must check terminal rules before calling."""
    score = 0
    phase = _phase(board)
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        score += sign * (
            _material_score(board, color)
            + _mobility_score(board, color)
            + _pst_score(board, color, phase)
            + _activity_score(board, color, phase)
        )

    return score
