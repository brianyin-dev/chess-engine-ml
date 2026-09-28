import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess

from backend.app import DEFAULT_BOOK_PATH, app
from engine.opening_book import OpeningBookError, choose_book_move, inspect_book


class OpeningBookTests(unittest.TestCase):
    def test_default_book_structure(self):
        info = inspect_book(DEFAULT_BOOK_PATH)
        self.assertEqual(info["bytes"], 486656)
        self.assertEqual(info["entries"], 30416)
        self.assertEqual(info["unique_position_keys"], 23813)

    def test_starting_choices_are_legal_and_reproducible(self):
        board = chess.Board()
        first = choose_book_move(board, DEFAULT_BOOK_PATH, rng=random.Random(7))
        second = choose_book_move(board, DEFAULT_BOOK_PATH, rng=random.Random(7))
        self.assertEqual(first, second)
        self.assertIn(first.move, board.legal_moves)
        self.assertGreater(first.weight, 0)
        self.assertGreater(first.alternatives, 1)

    def test_out_of_book_returns_none_without_mutation(self):
        board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
        before = board.fen()
        self.assertIsNone(choose_book_move(board, DEFAULT_BOOK_PATH))
        self.assertEqual(board.fen(), before)

    def test_bad_book_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            path.write_bytes(b"not a polyglot book")
            with self.assertRaises(OpeningBookError):
                choose_book_move(chess.Board(), path)
        with self.assertRaises(OpeningBookError):
            choose_book_move(chess.Board(), "/does/not/exist.bin")

    def test_api_uses_book_and_can_disable_it(self):
        client = app.test_client()
        response = client.post("/move", json={"moves": [], "time_ms": 20})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["source"], "book")
        self.assertIsNone(data["search"])
        self.assertEqual(data["book"]["file"], "gm2001.bin")
        self.assertIn(chess.Move.from_uci(data["move"]), chess.Board().legal_moves)

        response = client.post("/move", json={"moves": [], "time_ms": 20, "use_book": False})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["source"], "search")
        self.assertIsNone(data["book"])
        self.assertIsNotNone(data["search"])

    def test_api_respects_book_ply_limit(self):
        client = app.test_client()
        response = client.post("/move", json={"moves": ["e2e4"], "time_ms": 20,
                                               "max_book_ply": 1})
        self.assertEqual(response.get_json()["source"], "search")

    def test_api_validates_book_options(self):
        client = app.test_client()
        for payload in [{"use_book": "yes"}, {"max_book_ply": True},
                        {"max_book_ply": -1}, {"max_book_ply": 101}]:
            with self.subTest(payload=payload):
                self.assertEqual(client.post("/move", json=payload).status_code, 400)

    def test_api_reports_corrupt_configured_book(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            path.write_bytes(b"bad")
            with patch.dict("os.environ", {"CHESS_BOOK_PATH": str(path)}):
                response = app.test_client().post("/move", json={})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.get_json()["error"], "opening book error")


if __name__ == "__main__":
    unittest.main()
