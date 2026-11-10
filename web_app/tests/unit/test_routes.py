import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import main


class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_home_page_loads(self):
        with patch("main.discover_models", return_value={"en": {"small": {}}}):
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("FastAPI + HTMX", response.text)

    def test_home_page_handles_empty_registry(self):
        with patch("main.discover_models", return_value={}):
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("FastAPI + HTMX", response.text)

    def test_language_route_returns_empty_list_for_unknown_language(self):
        with patch("main.discover_models", return_value={"en": {"small": {}}}):
            response = self.client.get("/language?language=fr")

        self.assertEqual(response.status_code, 200)
        self.assertIn("<option", response.text)
