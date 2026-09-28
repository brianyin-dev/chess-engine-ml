import chess


class Board:
    """Thin wrapper around python-chess providing board state and move generation."""

    def __init__(self, fen: str = chess.STARTING_FEN):
        self._board = chess.Board(fen)

    def legal_moves(self):
        return list(self._board.legal_moves)

    def push(self, move: chess.Move):
        self._board.push(move)

    def pop(self):
        self._board.pop()

    def is_game_over(self) -> bool:
        return self._board.is_game_over()

    def result(self) -> str:
        return self._board.result()

    def turn(self) -> chess.Color:
        return self._board.turn

    def fen(self) -> str:
        return self._board.fen()

    def raw(self) -> chess.Board:
        """Access the underlying python-chess board directly."""
        return self._board

    def __repr__(self):
        return str(self._board)
