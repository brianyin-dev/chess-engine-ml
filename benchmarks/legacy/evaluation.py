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
PASSED_PAWN_BONUS = [0, 8, 14, 24, 40, 62, 0, 0]
REPETITION_PENALTY = 28
QUEEN_EARLY_DEVELOPMENT_PENALTY = 10

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

# Piece-square tables (from White's perspective, a1=index 0)
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


def _pst_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for piece_type, table in PST.items():
        for sq in board.pieces(piece_type, color):
            idx = sq if color == chess.WHITE else chess.square_mirror(sq)
            score += table[idx]
    return score


def _material_score(board: chess.Board, color: chess.Color) -> int:
    return sum(MATERIAL[pt] * len(board.pieces(pt, color)) for pt in MATERIAL)


def _mobility_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.color == color:
            score += MOBILITY_WEIGHTS.get(piece.piece_type, 0)
    return score


def _development_score(board: chess.Board, color: chess.Color) -> int:
    home_rank = 0 if color == chess.WHITE else 7
    score = 0

    for square in board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color):
        if chess.square_rank(square) != home_rank:
            score += DEVELOPMENT_BONUS

    queen_home = chess.D1 if color == chess.WHITE else chess.D8
    if board.piece_at(queen_home) is None:
        undeveloped_minors = 0
        for square in board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color):
            if chess.square_rank(square) == home_rank:
                undeveloped_minors += 1
        if undeveloped_minors >= 2:
            score -= QUEEN_EARLY_DEVELOPMENT_PENALTY

    return score


def _center_control_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for square in CENTER_SQUARES:
        if board.is_attacked_by(color, square):
            score += CENTER_CONTROL_BONUS
    return score


def _rook_activity_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    enemy = not color
    for square in board.pieces(chess.ROOK, color):
        file_index = chess.square_file(square)
        friendly_pawns = any(chess.square_file(pawn) == file_index for pawn in board.pieces(chess.PAWN, color))
        enemy_pawns = any(chess.square_file(pawn) == file_index for pawn in board.pieces(chess.PAWN, enemy))
        if not friendly_pawns and not enemy_pawns:
            score += ROOK_OPEN_FILE_BONUS
        elif not friendly_pawns:
            score += ROOK_SEMI_OPEN_FILE_BONUS
    return score


def _passed_pawn_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    enemy = not color
    for square in board.pieces(chess.PAWN, color):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        is_passed = True
        for enemy_pawn in board.pieces(chess.PAWN, enemy):
            enemy_file = chess.square_file(enemy_pawn)
            if abs(enemy_file - file_index) > 1:
                continue
            enemy_rank = chess.square_rank(enemy_pawn)
            if color == chess.WHITE and enemy_rank > rank_index:
                is_passed = False
                break
            if color == chess.BLACK and enemy_rank < rank_index:
                is_passed = False
                break
        if is_passed:
            progress = rank_index if color == chess.WHITE else 7 - rank_index
            score += PASSED_PAWN_BONUS[progress]
    return score


def _activity_score(board: chess.Board, color: chess.Color) -> int:
    return (
        _development_score(board, color)
        + _center_control_score(board, color)
        + _rook_activity_score(board, color)
        + _passed_pawn_score(board, color)
    )


def _repetition_score(board: chess.Board) -> int:
    if board.is_repetition(2):
        return -REPETITION_PENALTY if board.turn == chess.WHITE else REPETITION_PENALTY
    return 0


def evaluate(board: chess.Board) -> int:
    """
    Returns a centipawn score from White's perspective.
    Positive = White is better, negative = Black is better.
    """
    if board.is_checkmate():
        return -100_000 if board.turn == chess.WHITE else 100_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = _repetition_score(board)

    original_turn = board.turn
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        board.turn = color
        score += sign * _mobility_score(board, color)
        board.turn = original_turn
        score += sign * (
            _material_score(board, color)
            + _pst_score(board, color)
            + _activity_score(board, color)
        )

    board.turn = original_turn
    return score
