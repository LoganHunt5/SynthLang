import logging
import tempfile
import unittest
from pathlib import Path

import yaml

from pipeline.cli import run
from pipeline.trainer import Trainer


class RecordingTrainer(Trainer):
    def __init__(self):
        self.calls = []

    def train(self, config, *, warm_start_checkpoint=None):
        self.calls.append((config, warm_start_checkpoint))


class TestCli(unittest.TestCase):
    def _write_config(self, directory, warm_start=None):
        config = {
            "training": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 2,
            },
            "dataset": {
                "path": "data/train.csv",
                "formatter": "default",
                "meta_file_train": "metadata.csv",
            },
            "model": {"sample_rate": 16000},
        }
        if warm_start is not None:
            config["training"]["warm_start"] = warm_start

        config_path = Path(directory) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return config_path

    def test_run_starts_fresh_when_warm_start_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            trainer = RecordingTrainer()

            result = run(
                config_path,
                trainer_factory=lambda logger: trainer,
                logger=logging.getLogger("test_cli_fresh"),
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(trainer.calls), 1)
            self.assertIsNone(trainer.calls[0][1])

    def test_run_passes_warm_start_checkpoint_to_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                {
                    "enabled": True,
                    "checkpoint_path": "checkpoints/model.pth",
                },
            )
            trainer = RecordingTrainer()

            result = run(
                config_path,
                trainer_factory=lambda logger: trainer,
                logger=logging.getLogger("test_cli_warm"),
            )

            self.assertEqual(result, 0)
            self.assertEqual(
                trainer.calls[0][1],
                "checkpoints/model.pth",
            )

    def test_run_rejects_warm_start_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                {"enabled": True, "checkpoint_path": ""},
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint_path"):
                run(
                    config_path,
                    trainer_factory=lambda logger: RecordingTrainer(),
                    logger=logging.getLogger("test_cli_invalid"),
                )


if __name__ == "__main__":
    unittest.main()
