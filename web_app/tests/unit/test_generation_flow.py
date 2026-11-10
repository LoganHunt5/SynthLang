import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import main


class TestGenerationFlow(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_generate_rejects_unknown_language_or_model(self):
        with patch("main.discover_models", return_value={"en": {"small": {"model_path": "fake.pt", "config_path": "fake.json"}}}):
            response = self.client.post(
                "/generate",
                data={"language": "en", "model": "missing", "prompt": "hello world"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown language/model", response.text)

    def test_generate_rejects_blank_form_values(self):
        response = self.client.post(
            "/generate",
            data={"language": "", "model": "small", "prompt": "hello world"},
        )

        self.assertEqual(response.status_code, 422)

    def test_generate_happy_path_calls_synthesizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            fake_info = {
                "model_path": str(Path("/tmp/fake-model.pt")),
                "config_path": str(Path("/tmp/fake-config.json")),
            }
            with patch("main.discover_models", return_value={"en": {"small": fake_info}}), \
                 patch("main.synthesize") as mock_synthesize, \
                 patch.object(main, "AUDIO_DIR", audio_dir):
                response = self.client.post(
                    "/generate",
                    data={"language": "EN  ", "model": "small", "prompt": "hello world"},
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("/static/audio/", response.text)
            self.assertTrue(mock_synthesize.called)

