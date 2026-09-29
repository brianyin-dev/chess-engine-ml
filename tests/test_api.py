import unittest
import chess
from unittest.mock import patch
from backend.app import app, request_times


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
        self.assertEqual(data["game"]["over"], False)

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
        self.assertEqual(response.get_json()["game"],
                         {"over": True, "result": "1-0", "reason": "checkmate"})

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

    def test_public_request_bounds(self):
        self.assertEqual(self.client.post("/move", data=b"x" * 17_000,
                                          content_type="application/json").status_code, 413)
        response = self.client.post("/move", json={"moves": ["e2e4"] * 301})
        self.assertEqual(response.status_code, 400)
        self.assertIn("at most 300", response.get_json()["detail"])

    def test_busy_and_rate_limit_are_clear(self):
        with patch("backend.app.search_slots") as slots:
            slots.acquire.return_value = False
            response = self.client.post("/move", json={})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "engine busy")
        with patch("backend.app.RATE_LIMIT", 1):
            request_times.clear()
            first = self.client.post("/move", json={"depth": 1, "use_book": False})
            second = self.client.post("/move", json={"depth": 1, "use_book": False})
            request_times.clear()
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)


if __name__ == "__main__":
    unittest.main()
