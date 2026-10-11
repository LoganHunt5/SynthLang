import tempfile
import unittest
from pathlib import Path

import yaml

from training_pipeline.pipeline.config_manager import ConfigManager as config_manager


class TestConfigManager(unittest.TestCase):
    def _write_config(self, tmp_dir, config):
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return config_path

    def _valid_config(self):
        return {
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 10,
            },
            "dataset": {
                "path": "data/train.csv",
                "formatter": "default",
                "meta_file_train": "metadata.csv",
            },
            "model": {"sample_rate": 16000},
        }

    def test_load_config_returns_yaml_dict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                """
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10
dataset:
  path: data/train.csv
  max_samples: 50
  formatter: default
  meta_file_train: metadata.csv
model:
  sample_rate: 16000
""".strip()
            )

            manager = config_manager()
            result = manager.load_config(str(config_path))

            self.assertEqual(result["training"]["batch_size"], 32)
            self.assertEqual(result["model"]["sample_rate"], 16000)
            self.assertEqual(manager.model_config.sample_rate, 16000)
            self.assertEqual(manager.training_config.batch_size, 32)

    # tests with invaild value types
    def test_load_config_rejects_invalid_value_types(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                """
training:
  batch_size: large
  learning_rate: 0.001
  epochs: 10
dataset:
  path: data/train.csv
  formatter: default
  meta_file_train: metadata.csv
model:
  sample_rate: 16000
""".strip()
            )

            with self.assertRaisesRegex(RuntimeError, "batch_size must be an integer"):
                config_manager().load_config(str(config_path))
    
    # tests with malformed yaml 
    def test_load_config_rejects_invalid_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad.yaml"
            config_path.write_text("training: [bad")

            with self.assertRaisesRegex(RuntimeError, "Failed to load config"):
                config_manager().load_config(str(config_path))
    
    # tests for optional-field defaults
    def test_load_config_uses_defaults_for_optional_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(tmp_dir, self._valid_config())

            manager = config_manager()
            manager.load_config(str(config_path))

            self.assertEqual(manager.training_config.eval_batch_size, 1)
            self.assertEqual(manager.training_config.output_path, "")
            self.assertEqual(manager.training_config.eval_split_size, 0.0)
            self.assertEqual(manager.training_config.eval_split_max_size, 0)
            self.assertEqual(manager.model_config.type, "tacotron2")

    # tests for minimum boundary values
    def test_load_config_accepts_lower_boundary_values(self):
        config = self._valid_config()
        config["training"].update(
            batch_size=1,
            learning_rate=1e-12,
            epochs=1,
        )
        config["model"]["sample_rate"] = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(tmp_dir, config)

            self.assertIsInstance(config_manager().load_config(str(config_path)), dict)

    # tests for missing required sections
    def test_load_config_rejects_missing_required_sections(self):
        cases = (
            ("training", "batch_size must be an integer"),
            ("dataset", "dataset path must be provided"),
            ("model", "sample_rate must be an integer"),
        )

        for section, message in cases:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as tmp_dir:
                config = self._valid_config()
                del config[section]
                config_path = self._write_config(tmp_dir, config)

                with self.assertRaisesRegex(RuntimeError, message):
                    config_manager().load_config(str(config_path))

    # tests for zero and negative values
    def test_load_config_rejects_non_positive_values(self):
        cases = (
            ("batch_size", 0, "batch_size must be > 0"),
            ("learning_rate", 0, "learning_rate must be > 0"),
            ("epochs", 0, "epochs must be >= 1"),
            ("sample_rate", 0, "sample_rate must be > 0"),
        )

        for field, value, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp_dir:
                config = self._valid_config()
                section = "model" if field == "sample_rate" else "training"
                config[section][field] = value
                config_path = self._write_config(tmp_dir, config)

                with self.assertRaisesRegex(RuntimeError, message):
                    config_manager().load_config(str(config_path))

    # tests for boolean values used as numbers
    def test_load_config_rejects_boolean_numeric_values(self):
        for section, field in (
            ("training", "batch_size"),
            ("training", "learning_rate"),
            ("training", "epochs"),
            ("model", "sample_rate"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp_dir:
                config = self._valid_config()
                config[section][field] = True
                config_path = self._write_config(tmp_dir, config)

                with self.assertRaisesRegex(RuntimeError, field):
                    config_manager().load_config(str(config_path))

    # tests with unknown config keys
    def test_load_config_allows_unknown_keys(self):
        config = self._valid_config()
        config["training"]["future_option"] = "allowed"
        config["model"]["future_option"] = {"version": 2}

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(tmp_dir, config)

            self.assertIsInstance(config_manager().load_config(str(config_path)), dict)

    # tests for backend selection and model params
    def test_load_config_defaults_backend_and_model_type(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(tmp_dir, self._valid_config())
            manager = config_manager()
            manager.load_config(str(config_path))

        self.assertEqual(manager.backend, "coqui")
        self.assertEqual(manager.model_config.type, "tacotron2")
        self.assertEqual(manager.model_config.params, {})

    def test_load_config_reads_backend_and_model_params(self):
        config = self._valid_config()
        config["backend"] = "coqui"
        config["model"]["type"] = "tacotron2"
        config["model"]["params"] = {"r": 7, "attention_type": "original"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(tmp_dir, config)
            manager = config_manager()
            manager.load_config(str(config_path))

        self.assertEqual(manager.backend, "coqui")
        self.assertEqual(
            manager.model_config.params,
            {"r": 7, "attention_type": "original"},
        )

    def test_load_config_rejects_invalid_backend_and_params(self):
        config = self._valid_config()
        config["backend"] = ""
        config["model"]["type"] = 5
        config["model"]["params"] = ["r", 7]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(tmp_dir, config)

            with self.assertRaises(RuntimeError) as context:
                config_manager().load_config(str(config_path))

        message = str(context.exception)
        self.assertIn("backend must be a non-empty string", message)
        self.assertIn("model type must be a non-empty string", message)
        self.assertIn("model params must be a dictionary", message)


if __name__ == "__main__":
    unittest.main()
