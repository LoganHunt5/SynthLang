import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import main


class TestModelDiscovery(unittest.TestCase):
    def test_discover_models_finds_valid_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lang_dir = tmp_path / "en"
            model_dir = lang_dir / "tiny-model"
            model_dir.mkdir(parents=True)

            (model_dir / "config.json").write_text('{"model":"tiny"}', encoding="utf-8")
            (model_dir / "tiny-model.pt").write_bytes(b"fake checkpoint")

            with patch.object(main, "MODELS_DIR", tmp_path):
                registry = main.discover_models()

        self.assertIn("en", registry)
        self.assertIn("tiny-model", registry["en"])

    def test_discover_models_ignores_invalid_model_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lang_dir = tmp_path / "en"
            model_dir = lang_dir / "bad-model"
            model_dir.mkdir(parents=True)

            (model_dir / "config.json").write_text('{"model":"bad"}', encoding="utf-8")

            with patch.object(main, "MODELS_DIR", tmp_path):
                registry = main.discover_models()

        self.assertEqual(registry, {})

    def test_discover_models_returns_empty_when_models_dir_missing(self):
        with patch.object(main, "MODELS_DIR", Path("/definitely/missing/models")):
            registry = main.discover_models()

        self.assertEqual(registry, {})