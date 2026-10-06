import tempfile
import unittest
from pathlib import Path

import yaml

from pipeline.config_manager import ConfigManager


class TestWarmStartConfig(unittest.TestCase):
    def test_load_config_exposes_warm_start_options(self):
        config = {
            "training": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 2,
                "warm_start": {
                    "enabled": True,
                    "checkpoint_path": "checkpoints/model.pth",
                },
            },
            "dataset": {
                "path": "data/train.csv",
                "formatter": "default",
                "meta_file_train": "metadata.csv",
            },
            "model": {"sample_rate": 16000},
        }

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            manager = ConfigManager()

            manager.load_config(str(config_path))

            self.assertTrue(manager.training_config.warm_start_enabled)
            self.assertEqual(
                manager.training_config.warm_start_checkpoint_path,
                "checkpoints/model.pth",
            )

    def test_load_config_rejects_invalid_warm_start(self):
        config = {
            "training": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 2,
                "warm_start": {"enabled": "yes"},
            },
            "dataset": {
                "path": "data/train.csv",
                "formatter": "default",
                "meta_file_train": "metadata.csv",
            },
            "model": {"sample_rate": 16000},
        }

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))

            with self.assertRaisesRegex(RuntimeError, "warm_start.enabled"):
                ConfigManager().load_config(str(config_path))


if __name__ == "__main__":
    unittest.main()
