import tempfile
import unittest
from pathlib import Path

from training_pipeline.config_manager import ConfigManager as config_manager


class TestConfigManager(unittest.TestCase):
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

    def test_load_config_raises_runtime_error_for_bad_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad.yaml"
            config_path.write_text("training: [bad")

            with self.assertRaises(RuntimeError):
                config_manager().load_config(str(config_path))

    def test_config_validator_accepts_valid_config(self):
        valid_config = {
            "training": {"batch_size": 32, "learning_rate": 0.001, "epochs": 10},
            "dataset": {"path": "data/train.csv", "max_samples": 50},
            "model": {"sample_rate": 16000},
        }

        self.assertEqual(config_manager().config_validator(valid_config), [])

    def test_config_validator_rejects_non_dict_input(self):
        self.assertEqual(
            config_manager().config_validator(None),
            ["config must be a dictionary"],
        )
        self.assertEqual(
            config_manager().config_validator("not-a-dict"),
            ["config must be a dictionary"],
        )


if __name__ == "__main__":
    unittest.main()
