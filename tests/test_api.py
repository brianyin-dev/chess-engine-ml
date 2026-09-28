import unittest
import chess
from backend.app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_move_with_history_and_stats(self):
        response = self.client.post("/move", json={"moves": ["e2e4"], "depth": 2,
                                                    "time_ms": 500, "use_book": False})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        board = chess.Board()
        board.push_uci("e2e4")
        self.assertIn(chess.Move.from_uci(data["move"]), board.legal_moves)
        self.assertGreater(data["search"]["nodes"], 0)
        self.assertIsInstance(data["eval"], int)

    def test_bad_input_returns_400(self):
        for data in [[], None, {"depth": 0}, {"depth": "3"}, {"depth": True},
                     {"time_ms": -1}, {"time_ms": None}, {"time_ms": 10001},
                     {"moves": "e2e4"}, {"moves": [None]}, {"moves": ["e2e5"]},
                     {"fen": "not fen"}, {"fen": None},
                     {"fen": "8/8/8/8/8/8/8/8 w - - 0 1"}]:
            with self.subTest(data=data):
                self.assertEqual(self.client.post("/move", json=data).status_code, 400)

    def test_game_over(self):
        response = self.client.post("/move", json={"fen": "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "game over")

    def test_options(self):
        response = self.client.options("/move")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_frontend_is_served_by_backend(self):
        with self.client.get("/") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'id="chessboard"', response.data)
        with self.client.get("/chess.js") as response:
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
